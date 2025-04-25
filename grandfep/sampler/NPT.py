import logging
from typing import Union
from pathlib import Path
import math

import numpy as np
from mpi4py import MPI

from openmm import unit, app, openmm
from openmmtools.integrators import BAOABIntegrator
import parmed


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
        #: k\ :sub:`B`\ * T, with unit.
        self.kBT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        #: reference temperature of the system, with unit
        self.temperature: unit.Quantity = temperature

        # set up simulation
        #: The OpenMM Topology object. All the res_name, atom_index, atom_name, etc. are in this topology.
        self.topology = topology
        #: The OpenMM System object.
        self.system = system
        integrator = BAOABIntegrator(temperature, collision_rate, timestep)
        #: Simulation ties together Topology, System, Integrator, and Context in this sampler.
        self.simulation: app.Simulation = app.Simulation(self.topology, self.system, integrator, platform)

        # IO related
        #: Call this reporter to write the rst7 restart file.
        self.rst_reporter = parmed.openmm.reporters.RestartReporter(rst_file, 0, netcdf=True)
        #: Call this reporter to write the dcd trajectory file.
        self.dcd_reporter = None
        if dcd_file is not None:
            self.dcd_reporter = app.DCDReporter(dcd_file, 0)

        self.logger.info(f"T   = {temperature}.")
        self.logger.info(f"kBT = {self.kBT}.")

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

    def report_rst(self, state=None):
        """
        Write an Amber rst7 restart file.

        :return: None
        """
        if not state:
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

    def report_dcd(self, state=None):
        """
        Append a frame to the DCD trajectory file.

        :return: None
        """
        if not state:
            state = self.simulation.context.getState(getPositions=True)
        if not self.dcd_reporter:
            raise ValueError("DCD reporter is not set")
        self.dcd_reporter.report(self.simulation, state)
        self.logger.debug(f"Write DCD file {self.dcd_reporter.fname}")


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

    init_lambda_state : int
        Lambda state index for this replica, counting from 0

    lambda_dict : dict
        A dictionary of mapping from global parameters to their values in all the sampling states.


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
                 dcd_file: str = None,
                 init_lambda_state: int = None,
                 lambda_dict: dict = None
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
        #: Lambda state index for this replica, counting from 0
        self.init_lambda_state: int = None
        #: A dictionary of mapping from global parameters to their values in all the sampling states.
        self.lambda_dict: dict = None
        #: Index of the Lambda state which is simulated. All the init_lambda_state in this MPI run can be fetched with ``self.lambda_states_list[rank]``
        self.lambda_states_list: list = None
        #: Number of Lambda state to be sampled. It should be equal to the length of the values in ``lambda_dict``.
        self.n_lambda_states: int = None

        self.set_lambda_dict(init_lambda_state, lambda_dict)

        self.logger.info(f"Rank {self.rank} of {self.size} initialized NPT sampler")

    def set_lambda_dict(self, init_lambda_state: int, lambda_dict: dict) -> None:
        """
        Set internal attributes :
            - ``init_lambda_state``: (int) Lambda state index for this replica, counting from 0
            - ``lambda_dict``: (dict) Global parameter values for all the sampling states
            - ``lambda_states_list``: (list) Index of the Lambda state which is simulated. All the init_lambda_state in this MPI run
            - ``n_lambda_states``: (int) Number of Lambda state to be sampled

        You can have 10 states defined in the ``lambda_dict``, and only simulate 4 replicas. In this case,
        each value in the ``lambda_dict`` will have a length of 10, and ``n_lambda_states=10``,
        and the ``lambda_states_list`` will be a list of 4 integers, continuously increasing between 0 and 9,
        and init_lambda_state should be one of the values in ``lambda_states_list``.


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
        ##
        if init_lambda_state is None:
            raise ValueError("init_lambda_state should be set")
        if lambda_dict is None:
            raise ValueError("lambda_dict should be set")

        ## All values in the lambda_dict should have the same length
        self.n_lambda_states = len(lambda_dict[list(lambda_dict.keys())[0]])
        if len(set([len(v) for v in lambda_dict.values()])) != 1:
            raise ValueError("All values in the lambda_dict should have the same length")

        ## The length of the lambda_dict should no smaller than the number of replicas
        if self.n_lambda_states < self.size:
            raise ValueError("The length of the lambda_dict should no smaller than the number of replicas")

        ## The init_lambda_state should be smaller than the size of the lambda_dict
        if init_lambda_state >= self.n_lambda_states:
            raise ValueError("The init_lambda_state should be smaller than the size of the lambda_dict")

        ## Check if the global parameter exist in the context
        parameters = self.simulation.context.getParameters()
        for lam in lambda_dict:
            if lam not in parameters:
                raise ValueError(f"{lam} is not found as a Global Parameter")

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
        l_0 = self.lambda_states_list[0]
        for l_i in self.lambda_states_list[1:]:
            if l_i != l_0 + 1:
                raise ValueError(f"The lambda states are not continuously increasing: {self.lambda_states_list}")
            l_0 = l_i

        msg = "MPI rank              :" + "".join([f" {i:6d}" for i in range(self.size)])
        self.logger.info(msg)
        msg = "lambda_states         :" + "".join([f" {i:6d}" for i in self.lambda_states_list])
        self.logger.info(msg)
        for lambda_key, lambda_val_list in self.lambda_dict.items():
            msg = f"{lambda_key:<22}:" + "".join([f" {lambda_val_list[i]:6.3f}" for i in self.lambda_states_list])
            self.logger.info(msg)

        # set the current state
        self.logger.info(f"init_lambda_state={self.init_lambda_state}")
        for lam, val_list in self.lambda_dict.items():
            self.logger.info(f"Set {lam}={val_list[self.init_lambda_state]}")
            self.simulation.context.setParameter(lam, val_list[self.init_lambda_state])

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

        with open(log_input) as f:
            lines = f.readlines()
        re_step = 0
        for l in lines[-1::-1]:
            if "RE Step" in l:
                re_step = int(l.split("RE Step")[-1])
                break
        self.re_step = re_step + 1

    def _calc_neighbor_reduced_energy(self) -> np.array:
        """
        Use the current configuration and compute the energy of the neighboring sampling state. Later BAR can be performed on this data.

        Returns
        -------
        reduced_energy :
            reduced energy in kBT, no unit
        """
        reduced_energy = np.zeros(self.n_lambda_states, dtype=np.float64)
        state = self.simulation.context.getState(getEnergy=True)
        reduced_energy[self.init_lambda_state] = state.getPotentialEnergy() / self.kBT

        # when there is left neighbor
        if self.init_lambda_state >=1:
            i = self.init_lambda_state-1
            for lam, val_list in self.lambda_dict.items():
                self.simulation.context.setParameter(lam, val_list[i])
            state = self.simulation.context.getState(getEnergy=True)
            reduced_energy[i] = state.getPotentialEnergy() / self.kBT

        # when there is right neighbor
        if self.init_lambda_state < self.n_lambda_states-1:
            i = self.init_lambda_state +1
            for lam, val_list in self.lambda_dict.items():
                self.simulation.context.setParameter(lam, val_list[i])
            state = self.simulation.context.getState(getEnergy=True)
            reduced_energy[i] = state.getPotentialEnergy() / self.kBT

        # back to the current state
        for lam, val_list in self.lambda_dict.items():
            self.simulation.context.setParameter(lam, val_list[self.init_lambda_state])
        return reduced_energy

    def _calc_full_reduced_energy(self) -> np.array:
        """
        Use the current configuration and compute the energy of all the sampling states. Later MBAR can be performed on this data.

        Returns
        -------
        reduced_energy :
            reduced energy in kBT, no unit
        """
        reduced_energy = np.zeros(self.n_lambda_states, dtype=np.float64)
        state = self.simulation.context.getState(getEnergy=True)
        reduced_energy[self.init_lambda_state] = state.getPotentialEnergy() / self.kBT

        for i in range(self.n_lambda_states):
            if i != self.init_lambda_state:
                # set global parameters
                for lam, val_list in self.lambda_dict.items():
                    self.simulation.context.setParameter(lam, val_list[i])
                state = self.simulation.context.getState(getEnergy=True)
                reduced_energy[i] = state.getPotentialEnergy() / self.kBT

        # back to the current state
        for lam, val_list in self.lambda_dict.items():
            self.simulation.context.setParameter(lam, val_list[self.init_lambda_state])
        return reduced_energy

    def replica_exchange(self, calc_neighbor_only: bool = True):
        """
        Perform one neighbor swap replica exchange. In odd RE steps, attempt exchange between 0-1, 2-3, ...
        In even RE steps, attempt exchange between 1-2, 3-4, ... If RE is accepted, update the
        position/boxVector/velocity

        Parameters
        ----------
        calc_neighbor_only :
            If True, only the nearest neighbor will be calculated.

        Returns
        -------
        reduced_energy_matrix : np.array
            A numpy array with the size of (n_lambda_states, n_simulated_states)

        re_decision : dict
            The exchange decision between all the pairs. e.g., ``{(0,1):(True, ratio_0_1), (2,3):(True, ratio_2_3)}``.
            The key is the MPI rank pairs, and the value is the dicision and the acceptance ratio.

        """

        # re_step has to be the same across all replicas. If not, raise an error.
        re_step_all = self.comm.allgather(self.re_step)
        if len(set(re_step_all)) != 1:
            raise ValueError(f"RE step is not the same across all replicas: {re_step_all}")

        self.logger.info(f"RE Step {self.re_step}")

        # calculate reduced energy
        if calc_neighbor_only:
            reduced_energy = self._calc_neighbor_reduced_energy()
        else:
            reduced_energy = self._calc_full_reduced_energy() # kBT unit

        reduced_energy_matrix = np.empty((len(self.lambda_states_list), self.n_lambda_states),
                                         dtype=np.float64)
        self.comm.Allgather([reduced_energy,        MPI.DOUBLE],      # send buffer
                            [reduced_energy_matrix, MPI.DOUBLE])      # receive buffer

        # rank 0 decides the exchange
        if self.rank == 0:
            re_decision = {}
            for rank_i in range(self.re_step % 2, self.size - 1, 2):
                i = self.lambda_states_list[rank_i]
                delta_energy = (  reduced_energy_matrix[rank_i, i+1] + reduced_energy_matrix[rank_i + 1, i]
                                - reduced_energy_matrix[rank_i, i]   - reduced_energy_matrix[rank_i + 1, i+1])
                accept_prob = math.exp(-delta_energy)
                if np.random.rand() < accept_prob:
                    re_decision[(rank_i, rank_i + 1)] = (True, min(1,accept_prob))
                else:
                    re_decision[(rank_i, rank_i + 1)] = (False, min(1,accept_prob))
        else:
            re_decision = None  # placeholder on other ranks

        re_decision = self.comm.bcast(re_decision, root=0)

        # exchange position/boxVector/velocity
        for (rank_i, rank_j), (decision, ratio) in re_decision.items():
            # exchange boxVector/positions/velocities
            if decision:
                if self.rank == rank_i:
                    neighbor = rank_j
                elif self.rank == rank_j:
                    neighbor = rank_i
                else:
                    continue

                state = self.simulation.context.getState(getPositions=True, getVelocities=True)

                # Exchange with the neighbor
                vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
                vel = np.ascontiguousarray(vel, dtype='float64')
                recv_vel = np.empty_like(vel, dtype='float64')
                self.comm.Sendrecv(sendbuf=vel, dest=neighbor, sendtag=0, recvbuf=recv_vel, source=neighbor, recvtag=0)

                pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                pos = np.ascontiguousarray(pos, dtype='float64')
                recv_pos = np.empty_like(pos, dtype='float64')
                self.comm.Sendrecv(sendbuf=pos, dest=neighbor, sendtag=1, recvbuf=recv_pos, source=neighbor, recvtag=1)

                box_v = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
                box_v = np.ascontiguousarray(box_v, dtype='float64')
                recv_box_v = np.empty_like(box_v, dtype='float64')
                self.comm.Sendrecv(sendbuf=box_v, dest=neighbor, sendtag=2, recvbuf=recv_box_v, source=neighbor, recvtag=2)

                # set the new positions/boxVector/velocities
                self.simulation.context.setPeriodicBoxVectors(* (recv_box_v * unit.nanometer))
                self.simulation.context.setPositions(recv_pos * unit.nanometer)
                self.simulation.context.setVelocities(recv_vel * (unit.nanometer / unit.picosecond))

        # log results
        msg = ",".join([f"{i:13}" for i in reduced_energy])
        self.logger.info("Reduced Energy U_i(x): " + msg)

        if self.rank == 0:
            # log the exchange decision
            x_dict = {True: "x", False: " "}
            msg_ex = "Repl ex :"
            msg_pr = "Repl pr :"
            if not (0,1) in re_decision:
                msg_ex += f"{self.lambda_states_list[0]:2}   "
                msg_pr +=  "     "
            msg_ex += "   ".join([f"{self.lambda_states_list[j]:2} {x_dict[re_decision[(j, j+1)][0]]} {self.lambda_states_list[j+1]:2}" for j in range(0, self.size) if (j, j+1) in re_decision])
            msg_pr += "   ".join([f"{re_decision[(j, j+1)][1]:7.3f}" for j in range(0, self.size) if (j, j+1) in re_decision])
            if not (self.size-2, self.size-1) in re_decision:
                msg_ex += f"   {self.lambda_states_list[self.size - 1]:2}"
            self.logger.info(msg_ex)
            self.logger.info(msg_pr)


        self.re_step += 1
        return reduced_energy_matrix, re_decision

