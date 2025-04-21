from copy import deepcopy
import logging
from typing import Union
from pathlib import Path
import math

import numpy as np

from openmm import unit, app, openmm
from openmmtools.integrators import BAOABIntegrator

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
        the correct boxVector. Only rectangular box is supported.

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
        Chemical potential of the system. with units. Default is None.

    standard_volume :
        Standard volume of a water molecule in the reservoir. with units. Default is None.

    sphere_radius :
        Radius of the GCMC sphere. Default is 10.0 * unit.angstroms.

    reference_atoms :
        A list of atom indices in the topology that will be set as the center of the GCMC sphere. Default is None.


    Additional Attributes
    ---------------------
    gc_count : dict
        Dictionary to keep track of the GCMC moves. The keys are:
            - current_move: (int) the current move number
            - move: (list) list of move numbers
            - insert_delete: (list) list of 0/1 for insertion/deletion
            - work: (list) list of work values. In the unit of kcal/mol
            - box_GCMC: (list) list of 0/1 for box/GCMC
            - N: (list) list of number of water molecules in the box or GCMC sub-volume
            - accept: (list) list of 0/1 for rejection/acceptance

    chemical_potential : unit.Quantity
        Chemical potential of the system.

    standard_volume : unit.Quantity
        Standard volume of a water molecule in the reservoir.

    reference_atoms : list
        A list of atom indices in the topology that will be set as the center of the GCMC sphere.

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

        # counter for insertion/deletion, work, box/GCMC, N, accept/reject,
        self.gc_count = {
            "current_move"  : 0,
            "move"          : [],
            "insert_delete" : [],
            "work"          : [],
            "box_GCMC"      : [],
            "N"             : [],
            "accept"        : [],
        }

        self.chemical_potential = chemical_potential
        self.standard_volume = standard_volume

        self.simulation.context.setPositions(position)

        # set Adam value
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

        if self.sphere_radius > min(self.box_vectors[0, 0], self.box_vectors[1, 1], self.box_vectors[2, 2]) / 2:
            raise ValueError(f"The sphere radius {self.sphere_radius} is larger than half of the box size.")

        self.reference_atoms = reference_atoms
        self.logger.info(f"GCMC sphere is based on reference atom IDs: {self.reference_atoms}")
        at_list = [at for at in self.topology.atoms()]
        for at_index in self.reference_atoms:
            self.logger.info(f"    {at_list[at_index]}")


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



    def report(self, write_dcd=False):
        """
        Report the GC moves and write trajectory if needed.
        :param write_dcd:
        :return:
        """
        pass

    def insert_random_water_box(self, state: openmm.State, res_index: int):
        """
        Shift the coordinate+orientation of a water molecule to a random place in the box, or in the GCMC sphere.

        Parameters
        ----------
        state : openmm.State
            The current state of the simulation context. The positions will be read from this state.
        res_index : int
            The residue index of the water molecule to be shifted.

        Returns
        -------
        positions : unit.Quantity
            The new positions with the ghost water molecule shifted.
        """

        # Inside this function, all the positions are in nanometers.
        # Select a random position in the box
        x = np.random.uniform(0, self.box_vectors[0, 0].value_in_unit(unit.nanometer))
        y = np.random.uniform(0, self.box_vectors[1, 1].value_in_unit(unit.nanometer))
        z = np.random.uniform(0, self.box_vectors[2, 2].value_in_unit(unit.nanometer))
        insertion_point = np.array([x, y, z])

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

        return positions * unit.nanometer


    def gcmc_move(self, box: bool = 0):
        pass

    def insertion_move_box(self):
        """

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

        # change integrator

        # random (r,p) of the switching water molecule

        # work process
        ## Coulomb 0->1
        ## vdW 0->1

        # Acceptance ratio

        ## Accept :
        ### swap (r,p) of the switching water molecule with a ghost water molecule
        ### lambda_gc to 0
        ### set ghost to real, and update ghost_list

        ## Reject :
        ### revert the initial (r,p)
        ### lambda_gc to 0

        # update gc_count

        # change integrator back
        pass

    def deletion_move_box(self):
        pass

    def insertion_move_GCMC(self):
        pass

    def deletion_move_GCMC(self):
        pass




