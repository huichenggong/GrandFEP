from copy import deepcopy
import logging
from typing import Union
from pathlib import Path
import math

import numpy as np
from mpi4py import MPI

from openmm import unit, app, openmm
import parmed

from .. import utils

from .base import BaseGrandCanonicalMonteCarloSampler


class NoneqGrandCanonicalMonteCarloSampler(BaseGrandCanonicalMonteCarloSampler):
    """
    Nonequilibrium Grand Canonical Monte Carlo (Noneq-GCMC) sampler.

    In this class, GCMC is achieved by performing insertion/deletion in a nonequilibrium candidate Monte Carlo manner.
    The work value of the insertion/deletion will be used to evaluate the acceptance ratio. Insertion/deletion can
    either be performed in the whole box or in a sub-volume (active site) of the box. In an equilibrium sampling, when
    the water is free to move in/out of the sub-volume, I recommend alternating between GCMC and box.

    Parameters
    ----------
    system :
        The OpenMM System object. Must include `CustomNonbondedForce` and `NonbondedForce` with
        appropriate per-particle parameters and global parameter definitions.

    topology :
        The OpenMM Topology object. Must contain water molecules with the specified residue and atom names. Must have
        the correct boxVector. Only a rectangular box is supported.

    temperature :
        The reference temperature for the system, with proper units (e.g., kelvin).

    collision_rate :
        The collision rate (friction) for the Langevin integrator, with time units.

    timestep :
        The timestep for the integrator, with time units (e.g., femtoseconds).

    log :
        Path to the log file. This file will be opened in appended mode.

    platform :
        The OpenMM computational platform to use. Default is CUDA.

    water_resname :
        The residue name of water in the topology. Default is 'HOH'.

    water_O_name :
        The atom name of oxygen in water. Default is 'O'.

    position :
        Initial position of the system. Need to be provided for box Vectors. Default is None.

    chemical_potential :
        Chemical potential of the system, with units. Default is None.

    standard_volume :
        Standard volume of a water molecule in the reservoir. with units. Default is None.

    sphere_radius :
        Radius of the GCMC sphere. Default is 10.0 * unit.angstroms.

    reference_atoms :
        A list of atom indices in the topology that will be set as the center of the GCMC sphere. Default is None.

    rst_file :
        File name for the restart file.

    dcd_file :
        File name for the DCD trajectory file.


    """
    def __init__(self,
                 system: openmm.System,
                 topology: app.Topology,
                 temperature: unit.Quantity,
                 collision_rate: unit.Quantity,
                 timestep: unit.Quantity,
                 log: Union[str, Path],
                 platform: openmm.Platform = openmm.Platform.getPlatformByName('CUDA'),
                 water_resname: str = "HOH",
                 water_O_name: str = "O",
                 position: unit.Quantity = None,
                 chemical_potential=None,
                 standard_volume=None,
                 sphere_radius: unit.Quantity = 10.0*unit.angstroms,
                 reference_atoms: list =None,
                 rst_file: str = "md.rst7",
                 dcd_file: str = "md.dcd"
                 ):
        """
        """
        # safety check
        for val in [position, chemical_potential, standard_volume, sphere_radius]:
            if not isinstance(val, unit.Quantity):
                raise ValueError("position, chemical_potential and standard_volume must be provided as unit.Quantity.")

        if chemical_potential > 0.0 * unit.kilocalories_per_mole:
            raise ValueError(f"The chemical potsntial of water should be a negative value. {chemical_potential} is given")

        if reference_atoms is None:
            raise ValueError(f"reference_atoms should be provided")


        super().__init__(system, topology, temperature, collision_rate,
                         timestep, log, platform, water_resname, water_O_name)

        self.logger.info("Initializing NoneqGrandCanonicalMonteCarloSampler")

        """
        Dictionary to keep track of the GCMC moves. The keys are:
            - current_move: (int) the current move number
            - move: (list) list of move numbers
            - insert_delete: (list) list of 0/1 for insertion/deletion
            - work: (list) list of work values. In the unit of kcal/mol
            - box_GCMC: (list) list of 0/1 for box/GCMC
            - N: (list) list of number of water molecules in the box or GCMC sub-volume
            - accept: (list) list of 0/1 for rejection/acceptance
            - acc_prob: (list) list of float for acceptance probability
            """
        self.gc_count = {
            "current_move"  : 0,
            "move"          : [],
            "insert_delete" : [],
            "work"          : [],
            "box_GCMC"      : [],
            "N"             : [],
            "accept"        : [],
            "acc_prob"      : []
        }

        #: Chemical potential of the GC particle.
        self.chemical_potential: unit.Quantity = chemical_potential
        #: Standard volume of a water molecule in the reservoir.
        self.standard_volume: unit.Quantity = standard_volume

        self.simulation.context.setPositions(position)

        # Adam value settings
        self.sphere_radius = sphere_radius
        state = self.simulation.context.getState(getPositions=True)
        #: The box vectors of the system. The box has to be rectangular.
        self.box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                if not np.isclose(self.box_vectors[i, j]._value, 0.0):
                    raise ValueError("Only cuboidal box is supported.")
        volume_box = (self.box_vectors[0, 0] * self.box_vectors[1, 1] * self.box_vectors[2, 2] )
        self.Adam_box = self.chemical_potential/  self.kBT + math.log(volume_box / self.standard_volume)
        volume_GCMC = (4 * np.pi * self.sphere_radius ** 3) / 3
        self.Adam_GCMC = self.chemical_potential / self.kBT + math.log(volume_GCMC / self.standard_volume)
        self.logger.info(f"The box is {self.box_vectors[0, 0]} x {self.box_vectors[1, 1]} x {self.box_vectors[2, 2]} .")
        self.logger.info(f"The Adam value of the box is {self.Adam_box}.")
        self.logger.info(f"The Adam value of the GCMC sphere is {self.Adam_GCMC}.")

        # GCMC sphere settings
        if self.sphere_radius > min(self.box_vectors[0, 0], self.box_vectors[1, 1], self.box_vectors[2, 2]) / 2:
            raise ValueError(f"The sphere radius {self.sphere_radius} is larger than half of the box size.")
        #: A list of atom indices that will be set as the center of the GCMC sphere.
        self.reference_atoms = reference_atoms
        self.logger.info(f"GCMC sphere is based on reference atom IDs: {self.reference_atoms}")
        at_list = [at for at in self.topology.atoms()]
        for at_index in self.reference_atoms:
            self.logger.info(f"    {at_list[at_index]}")

        # IO related
        #: Call this reporter to write the rst7 restart file.
        self.rst_reporter = parmed.openmm.reporters.RestartReporter(rst_file, 0, netcdf=True)
        #: Call this reporter to write the dcd trajectory file.
        self.dcd_reporter = None
        if dcd_file is not None:
            self.dcd_reporter = app.DCDReporter(dcd_file, 0)

        self.logger.info(f"T   = {temperature}.")
        self.logger.info(f"kBT = {self.kBT}.")


    def update_gc_count(self,
                        insert_delete: int,
                        work: unit.Quantity,
                        box_GCMC: int,
                        n_water: int,
                        accept: bool,
                        acc_prob: float
                        ) -> None:
        """
        Update the internal `gc_count` dictionary that tracks GCMC insertion/deletion statistics.

        Parameters
        ----------
        insert_delete : int
            Indicator for the type of move: `0` for insertion, `1` for deletion.

        work : unit.Quantity
            Work value (e.g., from nonequilibrium switching), with units. The value in kcal/mol will be recorded.

        box_GCMC : int
            Region type indicator: `0` for the whole simulation box, `1` for a GCMC sub-volume.

        n_water : int
            Number of water molecules after this move in the whole box or in the GCMC sub-volume, depending on `box_GCMC`.

        accept : bool
            Indicator of whether the move was accepted

        acc_prob : float
            Acceptance probability.

        Returns
        -------
        None
        """
        self.gc_count["current_move"] += 1
        self.gc_count["move"].append(self.gc_count["current_move"])
        self.gc_count["insert_delete"].append(insert_delete)
        self.gc_count["work"].append(work.value_in_unit(unit.kilocalories_per_mole))
        self.gc_count["box_GCMC"].append(box_GCMC)
        self.gc_count["N"].append(n_water)
        if accept:
            self.gc_count["accept"].append(1)
        else:
            self.gc_count["accept"].append(0)
        self.gc_count["acc_prob"].append(acc_prob)


    def random_place_water(self, state: openmm.State, res_index: int, sphere_center: unit.Quantity=None) -> unit.Quantity:
        """
        Shift the coordinate+orientation of a water molecule to a random place. If the center is None, a random position
        in the box will be selected. If the center is not None, a randomed position in the GCMC sphere will be selected.

        Parameters
        ----------
        state : openmm.State
            The current state of the simulation context. The positions will be read from this state.
        res_index : int
            The residue index of the water molecule to be shifted.
        sphere_center : unit.Quantity
            The center of the GCMC sphere. If None, a random position in the box will be selected. Default is None.

        Returns
        -------
        positions : unit.Quantity
            The new positions with the water molecule shifted.
        velocities : unit.Quantity
            The new velocities with the water molecule shifted.
        """

        # Inside this function, all the positions are in nanometers.

        if sphere_center is None:
            # Select a random position in the box
            x = np.random.uniform(0, self.box_vectors[0, 0].value_in_unit(unit.nanometer))
            y = np.random.uniform(0, self.box_vectors[1, 1].value_in_unit(unit.nanometer))
            z = np.random.uniform(0, self.box_vectors[2, 2].value_in_unit(unit.nanometer))
            insertion_point = np.array([x, y, z])
        else:
            # random radius
            r = np.random.rand(1) ** (1/3)
            r *= self.sphere_radius.value_in_unit(unit.nanometer)
            # random direction
            v = np.random.normal(0, 1, 3)
            v /= np.linalg.norm(v)
            #
            insertion_point = sphere_center.value_in_unit(unit.nanometer) + r * v

        # Generate a random rotation matrix
        rot_matrix = utils.random_rotation_matrix()

        # Rotate and translate atoms except the 1st
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        o_index = self.water_res_2_atom[res_index][0]
        for atom_index in self.water_res_2_atom[res_index][1:]:
            relative_pos = positions[atom_index] - positions[o_index]
            new_relative = np.dot(rot_matrix, relative_pos)
            positions[atom_index] = insertion_point + new_relative

        # Translate the 1st atom
        positions[o_index] = insertion_point

        # rotate all velocities
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
        for atom_index in self.water_res_2_atom[res_index]:
            vel[atom_index, :] = np.dot(rot_matrix, vel[atom_index, :])

        return positions * unit.nanometer, vel * unit.nanometer/unit.picosecond


    def _move_init(self):
        """
        Initialize the GCMC move.

        Returns
        -------
        state : openmm.State
            The current state of the simulation context, with positions, velocities, and energy.

        pos_old : unit.Quantity
            The current positions of the system.

        vel_old : unit.Quantity
            The current velocities of the system.

        ghost_list_old : list
            The current list of ghost water molecules in the system.
        """
        state = self.simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=True,
        )
        pos_old = state.getPositions(asNumpy=True)
        vel_old = state.getVelocities(asNumpy=True)
        ghost_list_old = self.get_ghost_list()
        return state, pos_old, vel_old, ghost_list_old

    def work_process(self, n_prop: int, l_vdw_list: list, l_chg_list: list) -> tuple[bool, unit.Quantity, list]:
        """
        Perform the work process for insertion/deletion.

        Parameters
        ----------
        n_prop
            Number of propagation steps (equilibrium MD) between each lambda switching.

        l_vdw_list
            A list of value for ``lambda_gc_vdw``.

        l_chg_list
            A list of value for ``lambda_gc_coulomb``

        Returns
        -------
        explosion : bool
            Whether the non-equilibrium process get nan.

        protocol_work : unit.Quantity
            The work done during the insertion process.

        protocol_work_list : list
            A list of work done during each perturbation step, in kcal/mol. no unit in this list.
        """

        # change integrator
        self.compound_integrator.setCurrentIntegrator(1)
        protocol_work = 0.0 * unit.kilocalories_per_mole
        protocol_work_list = []
        explosion = False
        ## (vdW,chg), A -> B
        l_vdw = l_vdw_list[0]
        l_chg = l_chg_list[0]
        # work process
        self.simulation.context.setParameter("lambda_gc_vdw", l_vdw)
        self.simulation.context.setParameter("lambda_gc_coulomb", l_chg)
        for l_vdw, l_chg  in zip(l_vdw_list[1:-1], l_chg_list[1:-1]):
            # perturbation step
            energy_0 = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
            self.simulation.context.setParameter("lambda_gc_vdw", l_vdw)
            self.simulation.context.setParameter("lambda_gc_coulomb", l_chg)
            energy_i = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
            protocol_work += energy_i - energy_0
            protocol_work_list.append(
                (energy_i - energy_0).value_in_unit(unit.kilocalories_per_mole)
            )
            # check nan
            nan_flag0 = np.isnan(energy_0.value_in_unit(unit.kilocalories_per_mole))
            nan_flagi = np.isnan(energy_i.value_in_unit(unit.kilocalories_per_mole))
            if nan_flag0 or nan_flagi:
                explosion = True
                break

            # propagation step
            try:
                self.ncmc_integrator.step(n_prop)
            except:
                explosion = True
                break

            if not explosion:
                # last perturbation step
                l_vdw = l_vdw_list[-1]
                l_chg = l_chg_list[-1]
                energy_0 = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
                self.simulation.context.setParameter("lambda_gc_vdw", l_vdw)
                self.simulation.context.setParameter("lambda_gc_coulomb", l_chg)
                energy_i = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
                protocol_work += energy_i - energy_0
                protocol_work_list.append(
                    (energy_i - energy_0).value_in_unit(unit.kilocalories_per_mole)
                )
                # check nan
                nan_flag0 = np.isnan(energy_0.value_in_unit(unit.kilocalories_per_mole))
                nan_flagi = np.isnan(energy_i.value_in_unit(unit.kilocalories_per_mole))
                if nan_flag0 or nan_flagi:
                    explosion = True

        if explosion:
            self.logger.info(f"Insertion failed at (vdw,Coulomb)=({l_vdw},{l_chg})")

        # change integrator back
        self.compound_integrator.setCurrentIntegrator(0)

        self.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        self.simulation.context.setParameter("lambda_gc_coulomb", 0.0)

        return explosion, protocol_work, protocol_work_list

    def move_insertion_box(self, l_vdw_list: list, l_chg_list: list, n_prop: int) -> tuple[bool, float, unit.Quantity, list, int]:
        """
        Perform a non-equilibrium insertion, according the given lambda schedule

        Parameters
        ----------
        l_vdw_list : list
            A list of ``lambda_gc_vdw`` value that defines the path of insertion. It should start with 0.0 and end with 1.0.

        l_chg_list : list
            A list of ``lambda_gc_coulomb`` value that defines the path of insertion. It should start with 0.0 and end with 1.0.

        n_prop : int
            Number of propergation steps (equilibrium MD) between each lambda switching.

        Returns
        -------
        accept : bool
            Whether the insertion is accepted.

        acc_prob : float
            The acceptance probability of the insertion.

        protocol_work : unit.Quantity
            The work done during the insertion process, with unit.

        protocol_work_list : list
            A list of work values during each perturbation step, in kcal/mol. no unit in this list.

        n_water : int
            The number of water molecules in the system after the insertion.

        """
        assert np.isclose(l_vdw_list[0], 0)
        assert np.isclose(l_chg_list[0], 0)
        assert np.isclose(l_vdw_list[-1], 1)
        assert np.isclose(l_chg_list[-1], 1)
        self.logger.info("Insertion Box")
        # save initial (r,p)
        state, pos_old, vel_old, ghost_list_old = self._move_init()

        # random (r,p) of the ghost water, swap it with the switching water
        pos_new, vel_new = self.random_place_water(state, ghost_list_old[0])
        for at_ghost, at_switch in zip(self.water_res_2_atom[ghost_list_old[0]],
                                       self.water_res_2_atom[self.switching_water]):
            # swap
            pos_new[[at_ghost,at_switch]] = pos_new[[at_switch,at_ghost]]
            vel_new[[at_ghost,at_switch]] = vel_new[[at_switch,at_ghost]]
        self.simulation.context.setPositions(pos_new)
        self.simulation.context.setVelocities(vel_new)

        explosion, protocol_work, protocol_work_list = self.work_process(n_prop, l_vdw_list, l_chg_list)

        # Acceptance ratio
        #         num_of_water_including_s   - num_of_ghosts       -1
        n_water = len(self.water_res_2_atom) - len(ghost_list_old) -1 # The Total number of water mols include 1 switching water
        if explosion:
            acc_prob = 0
        else:
            acc_prob = math.exp(self.Adam_box) * math.exp(-protocol_work / self.kBT) / (n_water + 1)

        accept = np.random.rand() < acc_prob
        if accept:
            n_water += 1
            ### swap (r,p) of the switching water molecule with a ghost water molecule
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
            pos_new = state.getPositions(asNumpy=True)
            vel_new = state.getVelocities(asNumpy=True)
            for at_ghost, at_switch in zip(self.water_res_2_atom[ghost_list_old[0]],
                                           self.water_res_2_atom[self.switching_water]):
                # swap
                pos_new[[at_ghost,at_switch]] = pos_new[[at_switch,at_ghost]]
                vel_new[[at_ghost,at_switch]] = vel_new[[at_switch,at_ghost]]
            self.simulation.context.setPositions(pos_new)
            self.simulation.context.setVelocities(vel_new)

            ### set ghost to real, and update ghost_list
            ghost_list_old.pop(0)
            self.set_ghost_list(ghost_list_old, check_system=False)
        else:
            ### revert the initial (r,p)
            self.simulation.context.setPositions(pos_old)
            self.simulation.context.setVelocities(vel_old)

            ### move the selected ghost to the end of ghost_list
            self.ghost_list = self.ghost_list[1:] + [self.ghost_list[0]]

        # update gc_count
        self.update_gc_count(0, protocol_work, 0, n_water, accept, acc_prob)

        self.logger.info(f"GC_Insertion_Box Acc_prob={min(1, acc_prob):.3f}, Accept={accept}, N={n_water}")

        return accept, acc_prob, protocol_work, protocol_work_list, n_water

    def move_deletion_box(self, l_vdw_list: list, l_chg_list: list, n_prop: int) -> tuple[bool, float, unit.Quantity, list, int]:
        """
        Perform a non-equilibrium deletion, according the given lambda schedule

        Parameters
        ----------
        l_vdw_list : list
            A list of ``lambda_gc_vdw`` value that defines the path of insertion. It should start with 1.0 and end with 0.0.

        l_chg_list : list
            A list of ``lambda_gc_coulomb`` value that defines the path of insertion. It should start with 1.0 and end with 0.0.

        n_prop : int
            Number of propergation steps (equilibrium MD) between each lambda switching.

        Returns
        -------
        accept : bool
            Whether the insertion is accepted.

        acc_prob : float
            The acceptance probability of the insertion.

        protocol_work : unit.Quantity
            The work done during the insertion process, with unit.

        protocol_work_list : list
            A list of work values during each perturbation step, in kcal/mol. no unit in this list.

        n_water : int
            The number of water molecules in the system after the insertion.

        """
        assert np.isclose(l_vdw_list[0], 1)
        assert np.isclose(l_chg_list[0], 1)
        assert np.isclose(l_vdw_list[-1], 0)
        assert np.isclose(l_chg_list[-1], 0)
        self.logger.info("Deletion Box")
        # save initial (r,p)
        state, pos_old, vel_old, ghost_list_old = self._move_init()

        pos_new = deepcopy(pos_old)
        vel_new = deepcopy(vel_old)
        # random a real water to be deleted
        r_wat_set = set(self.water_res_2_atom) - set(ghost_list_old) - {self.switching_water}
        r_wat_index = np.random.choice(list(r_wat_set))
        self.set_ghost_list(ghost_list_old + [r_wat_index], check_system=False)
        for at_real, at_switch in zip(self.water_res_2_atom[r_wat_index],
                                       self.water_res_2_atom[self.switching_water]):
            # swap
            pos_new[[at_real,at_switch]] = pos_new[[at_switch,at_real]]
            vel_new[[at_real,at_switch]] = vel_new[[at_switch,at_real]]
        self.simulation.context.setPositions(pos_new)
        self.simulation.context.setVelocities(vel_new)

        explosion, protocol_work, protocol_work_list = self.work_process(n_prop, l_vdw_list, l_chg_list)

        # Acceptance ratio
        #         num_of_water_including_s   - num_of_ghosts       -1
        n_water = len(self.water_res_2_atom) - len(ghost_list_old) - 1  # The Total number of water mols include 1 switching water
        if explosion:
            acc_prob = 0
        else:
            acc_prob = math.exp(-self.Adam_box) * math.exp(-protocol_work / self.kBT) * n_water

        accept = np.random.rand() < acc_prob
        if accept:
            n_water -= 1

        else:
            ### revert the initial (r,p)
            self.simulation.context.setPositions(pos_old)
            self.simulation.context.setVelocities(vel_old)

            ### revert the ghost_list
            self.set_ghost_list(ghost_list_old, check_system=False)

        # update gc_count
        self.update_gc_count(1, protocol_work, 0, n_water, accept, acc_prob)

        self.logger.info(f"GC_Deletion_Box Acc_prob={min(1, acc_prob):.3f}, Accept={accept}, N={n_water}")

        return accept, acc_prob, protocol_work, protocol_work_list, n_water

    def get_sphere_center(self, positions: unit.Quantity) -> unit.Quantity:
        """
        Average the reference atoms positions

        Returns
        -------
        sphere_center : unit.Quantity
            The center of the GCMC sphere, with unit
        """
        center_pos = positions[self.reference_atoms]
        return np.mean(center_pos, axis=0)

    def get_water_state(self, positions) -> tuple[dict, np.array]:
        """
        Check whether the water molecule is inside the box (1) or not (0).

        Parameters
        ----------
        positions : unit.Quantity
            The current positions of the system.

        Returns
        -------
        water_state_dict : dict
            A dictionary label whether the water molecule is inside the box (1) or not (0).

        dist_all_o : np.array
            All the distance bewteen water oxygen and the center of the sphere.

        """
        sphere_center = self.get_sphere_center(positions).value_in_unit(unit.nanometer)
        simulation_box = np.array([self.box_vectors[0, 0].value_in_unit(unit.nanometer),
                                   self.box_vectors[1, 1].value_in_unit(unit.nanometer),
                                   self.box_vectors[2, 2].value_in_unit(unit.nanometer)])

        # shift everything so that the sphere is at [box_x/2, box_y/2, box_z/2]
        half_box = (simulation_box / 2)
        positions = positions.value_in_unit(unit.nanometer)
        positions += half_box - sphere_center
        positions = np.mod(positions, simulation_box)
        water_o_index = [o_index  for res_index, o_index in self.water_res_2_O.items()]
        dist_all_o = np.linalg.norm(positions[water_o_index] - half_box, axis=1)

        water_state_dict = {}
        for resid, dist in zip(self.water_res_2_O, dist_all_o):
            if dist < self.sphere_radius.value_in_unit(unit.nanometer):
                water_state_dict[resid] = 1
            else:
                water_state_dict[resid] = 0
        return water_state_dict, dist_all_o

    def move_insertion_GCMC(self, l_vdw_list: list, l_chg_list: list, n_prop: int) -> tuple[bool, float, unit.Quantity, list, int, bool]:
        """
        Perform a non-equilibrium insertion, according the given lambda schedule

        Parameters
        ----------
        l_vdw_list : list
            A list of ``lambda_gc_vdw`` value that defines the path of insertion. It should start with 0.0 and end with 1.0.

        l_chg_list : list
            A list of ``lambda_gc_coulomb`` value that defines the path of insertion. It should start with 0.0 and end with 1.0.

        n_prop : int
            Number of propergation steps (equilibrium MD) between each lambda switching.

        Returns
        -------
        accept : bool
            Whether the insertion is accepted.

        acc_prob : float
            The acceptance probability of the insertion.

        protocol_work : unit.Quantity
            The work done during the insertion process, with unit.

        protocol_work_list : list
            A list of work values during each perturbation step, in kcal/mol. no unit in this list.

        n_water : int
            The number of water molecules in the system after the insertion.

        """
        assert np.isclose(l_vdw_list[0], 0)
        assert np.isclose(l_chg_list[0], 0)
        assert np.isclose(l_vdw_list[-1], 1)
        assert np.isclose(l_chg_list[-1], 1)
        self.logger.info("Insertion GCMC Sphere")
        # save initial (r,p)
        state, pos_old, vel_old, ghost_list_old = self._move_init()

        # random (r,p) of the ghost water inside the sphere, swap it with the switching water
        sphere_center = self.get_sphere_center(pos_old)
        pos_new, vel_new = self.random_place_water(state, ghost_list_old[0], sphere_center)
        for at_ghost, at_switch in zip(self.water_res_2_atom[ghost_list_old[0]],
                                       self.water_res_2_atom[self.switching_water]):
            # swap
            pos_new[[at_ghost, at_switch]] = pos_new[[at_switch, at_ghost]]
            vel_new[[at_ghost, at_switch]] = vel_new[[at_switch, at_ghost]]
        self.simulation.context.setPositions(pos_new)
        self.simulation.context.setVelocities(vel_new)

        explosion, protocol_work, protocol_work_list = self.work_process(n_prop, l_vdw_list, l_chg_list)

        # Acceptance ratio
        state_new = self.simulation.context.getState(getPositions=True)
        water_stat_dict, _dist = self.get_water_state(state_new.getPositions(asNumpy=True))
        n_water = sum([w_state for res_index, w_state in water_stat_dict.items() if (res_index != self.switching_water) and (res_index not in ghost_list_old)])
        sw_water_inside = True
        if explosion:
            acc_prob = 0
        elif water_stat_dict[self.switching_water] == 0:
            acc_prob = 0
            sw_water_inside = False
        else:
            acc_prob = math.exp(self.Adam_GCMC) * math.exp(-protocol_work / self.kBT) / (n_water + 1) # 1 switching water

        accept = np.random.rand() < acc_prob
        if accept:
            n_water += 1
            ### swap (r,p) of the switching water molecule with a ghost water molecule
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
            pos_new = state.getPositions(asNumpy=True)
            vel_new = state.getVelocities(asNumpy=True)
            for at_ghost, at_switch in zip(self.water_res_2_atom[ghost_list_old[0]],
                                           self.water_res_2_atom[self.switching_water]):
                # swap
                pos_new[[at_ghost, at_switch]] = pos_new[[at_switch, at_ghost]]
                vel_new[[at_ghost, at_switch]] = vel_new[[at_switch, at_ghost]]
            self.simulation.context.setPositions(pos_new)
            self.simulation.context.setVelocities(vel_new)

            ### set ghost to real, and update ghost_list
            ghost_list_old.pop(0)
            self.set_ghost_list(ghost_list_old, check_system=False)
        else:
            ### revert the initial (r,p)
            self.simulation.context.setPositions(pos_old)
            self.simulation.context.setVelocities(vel_old)

            ### move the selected ghost to the end of ghost_list
            self.ghost_list = self.ghost_list[1:] + [self.ghost_list[0]]

            ### re-count the number of water inside sphere
            water_stat_dict, _dist = self.get_water_state(pos_old)
            n_water = sum([w_state for res_index, w_state in water_stat_dict.items() if
                           (res_index != self.switching_water) and (res_index not in self.ghost_list)])

        # update gc_count
        self.update_gc_count(0, protocol_work, 1, n_water, accept, acc_prob)

        self.logger.info(f"GC_Insertion_Sphere {min(1, acc_prob):.3f}, Accept={accept}, N={n_water}, Water_in_s={sw_water_inside}")

        return accept, acc_prob, protocol_work, protocol_work_list, n_water, sw_water_inside

    def move_deletion_GCMC(self, l_vdw_list: list, l_chg_list: list, n_prop: int) -> tuple[bool, float, unit.Quantity, list, int, bool]:
        """
        Perform a non-equilibrium deletion, according the given lambda schedule

        Parameters
        ----------
        l_vdw_list : list
            A list of ``lambda_gc_vdw`` value that defines the path of insertion. It should start with 1.0 and end with 0.0.

        l_chg_list : list
            A list of ``lambda_gc_coulomb`` value that defines the path of insertion. It should start with 1.0 and end with 0.0.

        n_prop : int
            Number of propergation steps (equilibrium MD) between each lambda switching.

        Returns
        -------
        accept : bool
            Whether the insertion is accepted.

        acc_prob : float
            The acceptance probability of the insertion.

        protocol_work : unit.Quantity
            The work done during the insertion process, with unit.

        protocol_work_list : list
            A list of work values during each perturbation step, in kcal/mol. no unit in this list.

        n_water : int
            The number of water molecules in the system after the insertion.

        """
        assert np.isclose(l_vdw_list[0], 1)
        assert np.isclose(l_chg_list[0], 1)
        assert np.isclose(l_vdw_list[-1], 0)
        assert np.isclose(l_chg_list[-1], 0)
        self.logger.info("Deletion GCMC Sphere")
        # save initial (r,p)
        state, pos_old, vel_old, ghost_list_old = self._move_init()

        pos_new = deepcopy(pos_old)
        vel_new = deepcopy(vel_old)
        # random a real water inside the sphere to be deleted
        water_stat_dict, _dist = self.get_water_state(pos_old)
        water_inside = []
        for res_index, state in water_stat_dict.items():
            flag_inside = state == 1
            flag_not_s  = res_index != self.switching_water
            flag_real   = res_index not in ghost_list_old
            if flag_inside and flag_not_s and flag_real:
                water_inside.append(res_index)
        n_water = len(water_inside)
        r_wat_index = np.random.choice(water_inside)
        self.set_ghost_list(ghost_list_old + [r_wat_index], check_system=False)
        for at_real, at_switch in zip(self.water_res_2_atom[r_wat_index],
                                      self.water_res_2_atom[self.switching_water]):
            # swap
            pos_new[[at_real, at_switch]] = pos_new[[at_switch, at_real]]
            vel_new[[at_real, at_switch]] = vel_new[[at_switch, at_real]]
        self.simulation.context.setPositions(pos_new)
        self.simulation.context.setVelocities(vel_new)

        explosion, protocol_work, protocol_work_list = self.work_process(n_prop, l_vdw_list, l_chg_list)

        state = self.simulation.context.getState(getPositions=True)
        water_stat_dict, _dist = self.get_water_state(state.getPositions(asNumpy=True))

        sw_water_inside = True
        if explosion:
            acc_prob = 0
        elif water_stat_dict[self.switching_water] == 0:
            acc_prob = 0
            sw_water_inside = False
        else:
            acc_prob = math.exp(-self.Adam_GCMC) * math.exp(-protocol_work / self.kBT) * n_water

        accept = np.random.rand() < acc_prob
        if accept:
            # re-count the number of water inside the sphere
            state_new = self.simulation.context.getState(getPositions=True)
            water_stat_dict, _dist = self.get_water_state(state_new.getPositions(asNumpy=True))
            n_water = sum([w_state for res_index, w_state in water_stat_dict.items() if
                           (res_index != self.switching_water) and (res_index not in self.ghost_list)])
        else:
            ### revert the initial (r,p)
            self.simulation.context.setPositions(pos_old)
            self.simulation.context.setVelocities(vel_old)

            ### revert the ghost_list
            self.set_ghost_list(ghost_list_old, check_system=False)

        return accept, acc_prob, protocol_work, protocol_work_list, n_water, sw_water_inside



class NoneqGrandCanonicalMonteCarloSamplerMPI(NoneqGrandCanonicalMonteCarloSampler):
    """
    Nonequilibrium Grand Canonical Monte Carlo (Noneq-GCMC) sampler with MPI (replica exchange) support.
    """
    def __init__(self,
                 system: openmm.System,
                 topology: app.Topology,
                 temperature: unit.Quantity,
                 collision_rate: unit.Quantity,
                 timestep: unit.Quantity,
                 log: Union[str, Path],
                 platform: openmm.Platform = openmm.Platform.getPlatformByName('CUDA'),
                 water_resname: str = "HOH",
                 water_O_name: str = "O",
                 position: unit.Quantity = None,
                 chemical_potential=None,
                 standard_volume=None,
                 sphere_radius: unit.Quantity = 10.0*unit.angstroms,
                 reference_atoms: list =None
                 ):
        """
        """
        pass