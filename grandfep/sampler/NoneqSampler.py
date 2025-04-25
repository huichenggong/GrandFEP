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
            - accept: (list) list of 0/1 for rejection/acceptance"""
        self.gc_count = {
            "current_move"  : 0,
            "move"          : [],
            "insert_delete" : [],
            "work"          : [],
            "box_GCMC"      : [],
            "N"             : [],
            "accept"        : [],
        }

        #: Chemical potential of the GC particle.
        self.chemical_potential: unit.Quantity = chemical_potential
        #: Standard volume of a water molecule in the reservoir.
        self.standard_volume: unit.Quantity = standard_volume

        self.simulation.context.setPositions(position)

        # Adam value settings
        self.sphere_radius = sphere_radius
        state = self.simulation.context.getState(getPositions=True)
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
                        accept: int) -> None:
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
            Number of water molecules in the whole box or in the GCMC sub-volume, depending on `box_GCMC`.

        accept : int
            Indicator of whether the move was accepted: `1` for accepted, `0` for rejected.

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
        self.gc_count["accept"].append(accept)


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

    def gcmc_move(self, box: bool = 0):
        pass

    def insertion_move_box(self, l_vdw_list: list, l_coulomb_list: list, n_prop: int):
        """
        Perform a non-equilibrium insertion. The vdw will be turned on first, followed by coulomb

        Parameters
        ----------
        l_vdw_list : list
            A list of ``lambda_gc_vdw`` value that defines the path of insertion

        l_coulomb_list : list
            A list of ``lambda_gc_coulomb`` value that defines the path of insertion

        n_prop : int
            Number of propergation step (equilibrium MD) between each lambda switching.

        Returns
        -------
        positions : unit.Quantity
            Positions after this GCMC move.
        velocities : unit.Quantity
            Velocities after this GCMC move.
        work : unit.Quantity
            Work during the insertion process.
        accept : bool
            Whether the insertion move was accepted.

        """
        # save initial (r,p)
        state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        pos_old = state.getPositions(asNumpy=True)
        vel_old = state.getVelocities(asNumpy=True)
        ghost_list_old = self.get_ghost_list()



        # random (r,p) of the switching water
        pos_new, vel_new = self.random_place_water(state, ghost_list_old[0])
        for at_ghost, at_switch in zip(self.water_res_2_atom[ghost_list_old[0]],
                                       self.water_res_2_atom[self.switching_water]):
            # swap
            pos_new[[at_ghost,at_switch]] = pos_new[[at_switch,at_ghost]]
            vel_new[[at_ghost,at_switch]] = vel_new[[at_switch,at_ghost]]
        self.simulation.context.setPositions(pos_new)
        self.simulation.context.setVelocities(vel_new)

        # work process
        self.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        self.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        # change integrator
        self.compound_integrator.setCurrentIntegrator(1)
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        ## vdW 0->1
        l_vdw = 0
        try:
            self.ncmc_integrator.step(n_prop)
        except:
            explosion = True
            self.logger.info(f"Insertion failed at lambda_gc_vdw={l_vdw}")
        for l_vdw in l_vdw_list[1:]:
            state = self.simulation.context.getState(getEnergy=True)
            energy_0 = state.getPotentialEnergy()
            self.simulation.context.setParameter("lambda_gc_vdw", l_vdw)
            state = self.simulation.context.getState(getEnergy=True)
            energy_i = state.getPotentialEnergy()
            protocol_work += energy_i - energy_0
            # check nan
            if np.isnan(energy_0.value_in_unit(unit.kilocalories_per_mole)) or np.isnan(
                    energy_i.value_in_unit(unit.kilocalories_per_mole)):
                explosion = True
                break
            try:
                self.ncmc_integrator.step(n_prop)
            except:
                explosion = True
                break
        if explosion:
            self.logger.info(f"Insertion failed at lambda_gc_vdw={l_vdw}")

        ## Coulomb 0->1, NonbondedForce has no addEnergyParameterDerivative
        if not explosion:
            for l_chg in l_coulomb_list[1:]:
                state = self.simulation.context.getState(getEnergy=True)
                energy_0 = state.getPotentialEnergy()
                self.simulation.context.setParameter("lambda_gc_coulomb", l_chg)
                state = self.simulation.context.getState(getEnergy=True)
                energy_i = state.getPotentialEnergy()
                protocol_work += energy_i - energy_0
                # check nan
                if np.isnan(energy_0.value_in_unit(unit.kilocalories_per_mole)) or np.isnan(
                        energy_i.value_in_unit(unit.kilocalories_per_mole)):
                    explosion = True
                    break
                try:
                    self.ncmc_integrator.step(n_prop)
                except:
                    explosion = True
                    break
            if explosion:
                self.logger.info(f"Insertion failed at lambda_gc_coulomb={l_chg}")


        # Acceptance ratio
        if explosion:
            acc_prob = 0
        else:
            n_water = len(self.water_res_2_atom) - len(ghost_list_old) + 1
            acc_prob = math.exp(self.Adam_box) * math.exp(-protocol_work / self.kBT) / len(self.water_res_2_atom)

        ## Accept :
        ### swap (r,p) of the switching water molecule with a ghost water molecule
        ### lambda_gc to 0
        ### set ghost to real, and update ghost_list

        ## Reject :
        ### revert the initial (r,p)
        ### lambda_gc to 0

        # update gc_count
        self.logger.info(f"Acceptance Ratio = {acc_prob}")

        # change integrator back
        self.compound_integrator.setCurrentIntegrator(0)

        #

    def deletion_move_box(self):
        pass

    def insertion_move_GCMC(self):
        pass

    def deletion_move_GCMC(self):
        pass




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