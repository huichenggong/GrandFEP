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
    system : openmm.System
        OpenMM system object, this system should include a barostat

    topology : app.Topology
        OpenMM topology object

    temperature : unit.Quantity
        Reference temperature of the system, with unit

    collision_rate : unit.Quantity
        Collision rate of the system, with unit. e.g., 1 / (1.0 * unit.picoseconds)

    timestep : unit.Quantity
        Timestep of the simulation, with unit. e.g., 4 * unit.femtoseconds with Hydrogen Mass Repartitioning

    log : Union[str, Path]
        Log file path for the simulation

    platform : openmm.Platform
        OpenMM platform to use for the simulation. Default is 'CUDA'.

    rst_file : str
        Restart file path for the simulation. Default is "md.rst7".

    dcd_file : str
        DCD file path for the simulation. Default is None, which means no dcd output.

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
                 dcd_file: str = None
                 ):
        """
        Initialize the NPT sampler
        """
        # prepare logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log)
        file_handler.setLevel(logging.INFO)
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
        self.simulation = app.Simulation(self.topology, self.system, integrator, platform)

        # IO related
        self.rst_reporter = parmed.openmm.reporters.RestartReporter(rst_file, 0, netcdf=True)
        if dcd_file is not None:
            self.dcd_reporter = app.DCDReporter(dcd_file, 0)
        else:
            self.dcd_reporter = None

    def check_temperature(self) -> unit.Quantity:
        """
        Check the reference temperature in the integrator and barostat. If they are not close, raise an Error

        Returns
        -------
        temperature : unit.Quantity
        """
        # find the barostat in system
        barostat = None
        for force in self.system.getForces():
            if isinstance(force, openmm.MonteCarloBarostat) or isinstance(force, openmm.MonteCarloMembraneBarostat):
                barostat = force
                break
        if barostat is None:
            raise ValueError("No barostat found in the system")
        ref_t_baro = barostat.getDefaultTemperature()
        if not np.isclose(ref_t_baro.value_in_unit(unit.kelvin), self.temperature.value_in_unit(unit.kelvin)):
            raise ValueError(f"Reference temperature in barostat ({ref_t_baro}) is not equal to the input temperature ({self.temperature})")
        return self.temperature

    def report_rst(self):
        """
        Write an Amber rst7 restart file.

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
        self.simulation.context.setPeriodicBoxVectors(*rst.getBoxVectors())
        self.simulation.context.setPositions(rst.getPositions())
        self.simulation.context.setVelocities(rst.getVelocities())
        self.logger.debug(f"Load boxVectors/positions/velocities from {rst_input}")


class NPTSamplerMPI(NPTSampler):
    """
    NPT Sampler class with MPI (replica exchange) support. Only Hamiltonian is allowed to be different.

    Parameters
    ----------
    system : openmm.System
        OpenMM system object, this system should include a barostat

    topology : app.Topology
        OpenMM topology object

    temperature : unit.Quantity
        Reference temperature of the system, with unit

    collision_rate : unit.Quantity
        Collision rate of the system, with unit. e.g., 1 / (1.0 * unit.picoseconds)

    timestep : unit.Quantity
        Timestep of the simulation, with unit. e.g., 4 * unit.femtoseconds with Hydrogen Mass Repartitioning

    log : Union[str, Path]
        Log file path for the simulation

    platform : openmm.Platform
        OpenMM platform to use for the simulation. Default is 'CUDA'.

    rst_file : str
        Restart file path for the simulation. Default is "md.rst7".

    dcd_file : str
        DCD file path for the simulation. Default is None, which means no dcd output.


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
                 dcd_file: str = None
                 ):
        """
        Initialize the NPT sampler with MPI support
        """
        super().__init__(system, topology, temperature, collision_rate, timestep, log, platform, rst_file, dcd_file)

        # MPI related properties
        #: MPI communicator
        self.comm = MPI.COMM_WORLD

        #: Rank of this process
        self.rank = self.comm.Get_rank()

        #: Size of the communicator (number of processes)
        self.size = self.comm.Get_size()

        # RE related properties
        #: Number of replica exchanges performed
        self.re_step = 0
        #: The lambda state for this replica, counting from 0
        self.init_lambda_state: int = None
        #: A dictionary of mapping from global parameters to their values in all the sampling states.
        self.lambda_dict: dict = None
        #: lambda state list in this set of simulation. The lambda state for each replica can be fetched with
        #: ``self.lambda_states_list[rank]``
        self.lambda_states_list: int = None

        self.logger.info(f"Rank {self.rank} of {self.size} initialized NPT sampler")

    def set_lambda_dict(self, init_lambda_state: int, lambda_dict: dict) -> None:
        """
        Set internal attributes :
            - ``init_lambda_state``: (int) Lambda state index for this replica, counting from 0
            - ``lambda_dict``: (dict) Global parameter values for all the sampling states
            - ``lambda_states_list``: (list) Lambda state indices in this set of simulation. The lambda state for each replica can be fetched with from this list.


        Parameters
        ----------
        init_lambda_state : int
            The lambda state for this replica, counting from 0

        lambda_dict : dict
            A dictionary of mapping from global parameters to their values in all the sampling states.
            For example, ``{"lambda_gc_vdw": [0.0, 0.5, 1.0, 1.0, 1.0], "lambda_gc_coulomb": [0.0, 0.0, 1.0, 0.5, 1.0]}``

        Returns
        -------
        None
        """
        # safety check
        ## All values in the lambda_dict should have the same length
        if len(set([len(v) for v in lambda_dict.values()])) != 1:
            raise ValueError("All values in the lambda_dict should have the same length")

        ## The length of the lambda_dict should no smaller than the number of replicas
        if len(lambda_dict[list(lambda_dict.keys())[0]]) < self.size:
            raise ValueError("The length of the lambda_dict should no smaller than the number of replicas")

        ## The init_lambda_state should be smaller than the size of the lambda_dict
        if init_lambda_state >= len(lambda_dict[list(lambda_dict.keys())[0]]):
            raise ValueError("The init_lambda_state should be smaller than the size of the lambda_dict")

        ## lambda_dict should be identical across all MPI
        all_l_dict = self.comm.allgather(lambda_dict)
        for l_dict in all_l_dict[1:]:
            if l_dict.keys() != all_l_dict[0].keys():
                raise ValueError(f"The lamda in all replicas are not the same \n{all_l_dict[0].keys()}\n{l_dict.keys()}")
            for lam_str in all_l_dict[0]:
                val_list0 = all_l_dict[0][lam_str]
                val_listi = l_dict[lam_str]
                if not np.all(np.isclose(val_list0, val_listi)):
                    raise ValueError(f"{lam_str} is not the same across all replicas \n{val_list0}\n{val_listi}")

        self.init_lambda_state = init_lambda_state
        self.lambda_dict = lambda_dict
        # all gather the lambda states
        self.lambda_states_list = self.comm.allgather(self.init_lambda_state)
        msg = "MPI rank              :" + "".join([f" {i:6d}" for i in range(self.size)])
        self.logger.info(msg)
        msg = "lambda_states         :" + "".join([f" {i:6d}" for i in self.lambda_states_list])
        self.logger.info(msg)
        for lambda_key, lambda_val_list in self.lambda_dict.items():
            msg = f"{lambda_key:<22}:" + "".join([f" {lambda_val_list[i]:6.3f}" for i in self.lambda_states_list])
            self.logger.info(msg)

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

    def replica_exchange(self, calc_neighbor_only: bool = True):
        """
        Perform one neighbor swap replica exchange. In odd RE steps, attempt exchange between 0-1, 2-3, ...
        In even RE steps, attempt exchange between 1-2, 3-4, ... If RE is accepted, update the
        position/boxVector/velocity

        Parameters
        ----------
        """
        pass