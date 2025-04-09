from copy import deepcopy
import logging

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
    the water is free to move in/out of the sub-volume, I recommend using both the whole box and the sub-volume for
    certain ratio.
    """
    def __init__(self, system, topology, temperature,
                 collision_rate, timestep,
                 log,
                 platform=openmm.Platform.getPlatform('CUDA'),
                 water_resname="HOH", water_O_name="O"):
        """
        Initialize the NoneqGCMC sampler.
        :param system:
        :param topology:
        :param temperature:
        :param collision_rate:
        :param timestep:
        :param log:
        :param platform:
        :param water_resname:
        :param water_O_name:
        """
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

    def update_gc_count(self, insert_delete, work, box_GCMC, n_water, accept):
        """
        Update the gc_count dictionary.

        :param insert_delete: (int)

            0 for insertion, 1 for deletion

        :param work: (unit.quantity.Quantity)

            work value, with unit. value in kcal/mol will be saved
            
        :param box_GCMC: (int)

            0 for whole box, 1 for sub-volume
            
        :param n_water: (int)

            number of water molecules in the whole box or in the GCMC sub-volume
            
        :param accept: (int)

            0 for rejection, 1 for acceptance
        """
        self.gc_count["current_move"] += 1
        self.gc_count["move"].append(self.gc_count["current_move"])
        self.gc_count["insert_delete"].append(insert_delete)
        self.gc_count["work"].append(work.value_in_unit(unit.kilocalories_per_mole))
        self.gc_count["box_GCMC"].append(box_GCMC)
        self.gc_count["N"].append(n_water)
        self.gc_count["accept"].append(accept)
