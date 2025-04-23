from copy import deepcopy
import logging
from typing import Union
from pathlib import Path
import math

import numpy as np

from openmm import unit, app, openmm

from .. import utils

from .NoneqSampler import NoneqGrandCanonicalMonteCarloSampler

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