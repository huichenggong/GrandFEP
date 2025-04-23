from copy import deepcopy
import logging
from typing import Union
from pathlib import Path

import numpy as np
from mpi4py import MPI

from openmm import unit, app, openmm
from openmmtools.integrators import BAOABIntegrator
import parmed

from .. import utils


class NPTSampler:
    """
    NPT Sampler class

    Parameters
    ----------
    """
    def __init__(self,
                 system: openmm.System,
                 topology: app.Topology,
                 temperature: unit.Quantity,
                 collision_rate: unit.Quantity,
                 timestep: unit.Quantity,
                 log: Union[str, Path],
                 platform: openmm.Platform = openmm.Platform.getPlatformByName('CUDA'),
                 rst_file: str = "md.rst7",
                 dcd_file: str = "md.dcd"
                 ):
        """
        Initialize the NPT sampler
        """
        # prepare logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                                    "%m-%d %H:%M:%S"))
        self.logger.addHandler(file_handler)
        self.logger.info("Initializing NPT Sampler")

        # constants and simulation configuration
        self.kBT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.temperature = temperature

        # set up simulation
        self.topology = topology
        self.system = system
        integrator = BAOABIntegrator(temperature, collision_rate, timestep)
        self.simulation = app.Simulation(self.topology, self.system, platform)

        # IO related
        self.rst_reporter = parmed.openmm.reporters.RestartReporter(rst_file, 0, netcdf=True)
        self.dcd_reporter = app.dcdreporter.DCDReporter(dcd_file, 0)

    def check_temperature(self) -> unit.Quantity:
        """
        Check the reference temperature in the integrator and barostat. If they are not close, raise an Error

        Returns
        -------
        temperature : unit.Quantity
        """
        pass

    def report_rst(self):
        """
        Write a Amber rst7 restart file.

        :return: None
        """
        state = self.simulation.context.getState(getPositions=True, getVelocities=True)
        self.rst_reporter.report(self.simulation, state)
        self.logger.debug(f"Write restart file {self.rst_reporter.fname}")

    def load_rst(self, rst_input: Union[str, Path]):
        """
        Load positions/boxVector/velocities from a restart file

        Parameters
        ----------
        rst_input : Union[str, Path]
            Amber inpcrd file

        Returns
        -------
        None
        """
        rst = app.AmberInpcrdFile(rst_input)
        sim.context.setPeriodicBoxVectors(*rst.getBoxVectors())
        sim.context.setPositions(rst.getPositions())
        sim.context.setVelocities(rst.getVelocities())
        self.logger.debug(f"Load boxVectors/positions/velocities from {rst_input}")


class NPTSamplerMPI:
    """
    NPT Sampler class with MPI (replica exchange) support. Only Hamiltonian is allowed to be different.
    """
    def __init__(self,
                 system: openmm.System,
                 topology: app.Topology,
                 temperature: unit.Quantity,
                 collision_rate: unit.Quantity,
                 timestep: unit.Quantity,
                 log: Union[str, Path],
                 platform: openmm.Platform = openmm.Platform.getPlatformByName('CUDA'),
                 rst_file: str = "md.rst7",
                 dcd_file: str = "md.dcd"
                 ):
        """
        Initialize the NPT sampler with MPI support
        """
        super().__init__(system, topology, temperature, collision_rate, timestep, log)
        self.re_step = 0
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def set_re_step(self, re_step: int):
        """
        Parameters
        ----------
        re_step : int
            Number of replica exchanges to perform

        Returns
        -------
        None
        """
        self.re_step = re_step

    def set_re_step_from_log(self, log_input: Union[str, Path]):
        """
        Read the previous log file and determine the RE step.

        Parameters
        ----------
        log_input : Union[str, Path]
            previous log file where the RE step counting should continue.

        Returns
        -------
        None
        """

    def replica_exchange(self, calc_neighbor_only: bool = True, lambda_dict: dict = None):
        """
        Perform one neighbor swap replica exchange. In odd RE steps, attempt exchange between 0-1, 2-3, ...
        In even RE steps, attempt exchange between 1-2, 3-4, ... If RE is accepted, update the
        position/boxVector/velocity

        Parameters
        ----------
        """
        pass