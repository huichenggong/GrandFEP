from copy import deepcopy
import logging
from typing import Union
from pathlib import Path
import math

import numpy as np

from openmm import unit, app, openmm
from openmmtools.integrators import BAOABIntegrator

from .. import utils

class BaseGrandCanonicalMonteCarloSampler:
    """
    Base class for Grand Canonical Monte Carlo (GCMC) sampling.

    This class provides a flexible framework for customizing OpenMM forces so that water molecules
    can be alchemically inserted (ghost → real) or deleted (real → ghost). Each water is assigned two properties:

    - `is_real`:
        - 1.0 : real water
        - 0.0 : ghost water.
    - `is_switching`:
        - 1.0 for the switching water (used for NEQ insertion/deletion)
        - 0.0 for all other waters.

    The last water in the system is selected as the switching water (is_real=1, is_switching=1).
    During an insertion or deletion attempt, this water is swapped (coordinate and velocity)
    with a ghost or real water, enabling NEQ perturbation moves.

    vdW interactions are handled via `self.custom_nonbonded_force` (`openmm.CustomNonbondedForce`), where
    per-particle parameters `is_real` and `is_switching` interact with the global parameter ``lambda_gc_vdw``.

    Electrostatic interactions are handled by `self.nonbonded_force` (`openmm.NonbondedForce`). Ghost waters
    are given zero charge, and switching waters use `ParticleParameterOffset` with a global parameter
    ``lambda_gc_coulomb``.

    Parameters
    ----------
    system :
        The OpenMM System object. Must include `CustomNonbondedForce` and `NonbondedForce` with
        appropriate per-particle parameters and global parameter definitions.

    topology :
        The OpenMM Topology object. Must contain water molecules with the specified residue and atom names.

    temperature :
        The reference temperature for the system, with proper units (e.g., kelvin).

    collision_rate :
        The collision rate (friction) for the Langevin integrator, with time units.

    timestep :
        The timestep for the integrator, with time units (e.g., femtoseconds).

    log :
        Path to the log file. This file will be opened in append mode.

    platform :
        The OpenMM computational platform to use. Default is CUDA.

    water_resname :
        The residue name of water in the topology. Default is 'HOH'.

    water_O_name :
        The atom name of oxygen in water. Default is 'O'.

    create_simulation :
        Whether to create a system inside this class. When you only want to customize the system using this class,
        you can set this to False, to avoid unecessary memory usage.

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
                 create_simulation: bool = True,
                 ):
        """
        """

        # prepare logger
        #: Logger for the Sampler
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:  # Avoid adding multiple handlers
            # remove the existing handlers
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
                # close the handler
                handler.close()
        file_handler = logging.FileHandler(log)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                                    "%m-%d %H:%M:%S"))
        self.logger.addHandler(file_handler)
        self.logger.info("Initializing BaseGrandCanonicalMonteCarloSampler")

        #: The OpenMM System object.
        self.system: openmm.System = None
        #: 2 integrators in this attribute. The 1st one is for Canonical simulation, the 2nd one is for non-equilibirum insertion/deletion.
        self.compound_integrator: openmm.CompoundIntegrator = None
        #: Simulation ties together Topology, System, Integrator, and Context in this sampler.
        self.simulation: app.Simulation = None
        #: The OpenMM Topology object. All the res_name, atom_index, atom_name, etc. are in this topology.
        self.topology: app.Topology = topology

        # constants and simulation configuration

        #: k\ :sub:`B`\ * T, with unit.
        self.kBT: unit.Quantity = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        #: The reference temperature for the system, with proper units (e.g., kelvin).
        self.temperature: unit.Quantity = temperature
        #: Adam value, considering the whole simulation box and all water molecules.
        self.Adam_box: unit.Quantity = None
        #: Adam value, considering the selected sphere and water molecules in the sphere.
        self.Adam_GCMC: unit.Quantity = None


        #: A list of residue indices of water that are set to ghost. Should only be modified with set_ghost_list(), and get by get_ghost_list().
        self.ghost_list = []
        #: A dictionary of residue index to a list of atom indices of water.
        self.water_res_2_atom: dict = None
        #: A dictionary of residue index to the atom index of water oxygen.
        self.water_res_2_O: dict = None
        #: The residue index of the switching water. The switching water will be set as the last water during initialization.
        #: It should not be changed during the simulation, as the ParticleParameterOffset can not be updated in NonbondedForce.
        self.switching_water: int = None
        #: The number of points in the water model. 3 for TIP3P, 4 for OPC.
        self.num_of_points_water: int = None

        # system and force field related attributes

        #: These force(s) handles vdW. They have PerParticleParameter is_real and is_switching to control real/ghost and
        #: switching/non-switching to identify the only switching water. It also has a global parameter
        # ``lambda_gc_vdw`` to control the intraction of the switching water.
        self.custom_nonbonded_force_list: list[openmm.CustomNonbondedForce] = []
        #: This force handles Coulomb. The switching water has ParticleParameterOffset with global parameter
        #: ``lambda_gc_coulomb`` to control the switching water.
        self.nonbonded_force: openmm.NonbondedForce = None
        #: A dictionary to track the nonbonded parameter of water. The keys are ``charge``, ``sigma``, ``epsilon``,
        #: The values are a list of parameters with unit.
        self.wat_params: dict = None
        #: The type of the system. Can be Amber, Charmm or Hybrid depending on the system and energy expression in the given CustomNonbondedForce.
        self.system_type: str = None

        # preparation based on the topology
        self.logger.info("Check topology")
        self.num_of_points_water = self._check_water_points(water_resname)
        self._find_all_water(water_resname, water_O_name)

        # preparation based on the system and force field
        self.logger.info("Prepare system")
        self.system_type = self._check_system(system)
        self._get_water_parameters(water_resname, system)

        if self.system_type == "Amber":
            self.customise_force_amber(system)
        elif self.system_type == "Charmm":
            self.customise_force_charmm(system)
        elif self.system_type == "Hybrid":
            self.customise_force_hybrid(system)
        else:
            raise ValueError(f"The system ({self.system_type}) cannot be customized. Please check the system.")

        if create_simulation:
            # preparation of integrator, simulation
            self.logger.info("Prepare integrator and simulation")
            self.compound_integrator = openmm.CompoundIntegrator()
            integrator = BAOABIntegrator(temperature, collision_rate, timestep)
            self.compound_integrator.addIntegrator(integrator) # for EQ run
            self.ncmc_integrator = BAOABIntegrator(temperature, collision_rate, timestep)
            self.compound_integrator.addIntegrator(self.ncmc_integrator) # for NCMC run

            self.simulation = app.Simulation(self.topology, self.system, self.compound_integrator, platform)
            self.compound_integrator.setCurrentIntegrator(0)

            self.logger.info(f"T   = {temperature}.")
            self.logger.info(f"kBT = {self.kBT}.")

    def _find_all_water(self, resname, water_O_name):
        """
        Check topology setup

            self.water_res_2_atom

            self.water_res_2_O

            self.switching_water

        :param resname: (str)
            The residue name of water in the topology.
        :param water_O_name: (str)
            The atom name of oxygen in the water.

        :return: None
        """
        # check if the system has water
        self.water_res_2_atom = {}
        self.water_res_2_O = {}
        for res in self.topology.residues():
            if res.name == resname:
                self.switching_water = res.index
                self.water_res_2_atom[res.index] = []
                self.water_res_2_O[res.index] = None
                for atom in res.atoms():
                    self.water_res_2_atom[res.index].append(atom.index)
                    if atom.name == water_O_name:
                        self.water_res_2_O[res.index] = atom.index
                if self.water_res_2_O[res.index] is None:
                    raise ValueError(f"The water ({resname}) does not have the oxygen atom ({water_O_name}). Please check the topology.")
        self.logger.info(f"Water res_index={self.switching_water} will be set as the switching water")

        if len(self.water_res_2_atom) == 0:
            raise ValueError(f"The topology does not have any water({resname}). Please check the topology.")

    def _check_water_points(self, resname):
        """
        Check if the water model is 3-point or 4-point in topology.
        :return: int
        """
        # check if the system has water
        for res in self.topology.residues():
            if res.name == resname:
                return len(list(res.atoms()))
        return 0

    def _check_system(self, system):
        """
        Check if system can be converted to a GCMC system.

        :param system: (openmm.System)
            The system to be checked.
        :return: (str)
            The type of the system. Can be Amber, Charmm or Hybrid
            Amber should have NonbondedForce without CustomNonbondedForce
            Charmm should have NonbondedForce and CustomNonbondedForce. The EnergyFunction that is allowed in
            CustomNonbondedForce is `(a/r6)^2-b/r6; r6=r^6;a=acoef(type1, type2);b=bcoef(type1, type2)` or
            `acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6;`

            Hybrid should have
                CustomBondForce            For perturbed bonds
                HarmonicBondForce
                CustomAngleForce           For perturbed angles
                HarmonicAngleForce
                CustomTorsionForce         For perturbed dihedrals
                PeriodicTorsionForce
                NonbondedForce             For perturbed Coulomb
                CustomNonbondedForce       For perturbed Lennard-Jones
                CustomBondForce_exceptions For perturbed exceptions, mostly 1-4 nonbonded
        """
        force_name_list = [f.getName() for f in system.getForces()]
        c_bond_flag     = "CustomBondForce" in force_name_list
        c_angle_flag    = "CustomAngleForce" in force_name_list
        c_torsion_flag  = "CustomTorsionForce" in force_name_list
        nb_force_flag   = "NonbondedForce" in force_name_list
        c_nb_force_flag = "CustomNonbondedForce" in force_name_list

        # all True, Hybrid
        system_type = None
        if c_bond_flag and c_angle_flag and c_torsion_flag and nb_force_flag and c_nb_force_flag:
            system_type =  "Hybrid"
        # NonbondedForce only, Amber
        elif nb_force_flag and not c_bond_flag and not c_angle_flag and not c_torsion_flag and not c_nb_force_flag:
            system_type =  "Amber"
        # NonbondedForce and CustomNonbondedForce, nothing else, Charmm
        elif nb_force_flag and c_nb_force_flag and not c_bond_flag and not c_angle_flag:
            system_type =  "Charmm"
        else:
            msg = "Here are the forces in the system: \n"
            for f in system.getForces():
                msg += f"{f.getName()}\n"
            msg += "The system is not supported. Please check the force in the system."
            raise ValueError(msg)

        # Other checks
        for f in system.getForces():
            # 1. PME should be used in nonbondedForce
            if f.getName() == "NonbondedForce":
                if f.getNonbondedMethod() != openmm.NonbondedForce.PME:
                    raise ValueError("PME should be used for long range electrostatics")

            # 2. Barostat should not be used
            if f.getName() == "MonteCarloBarostat":
                raise ValueError("Barostat should not be used in GCMC simulation")

        return system_type

    def _get_water_parameters(self, resname, system):
        """
        Get the charge of water and save it in self.wat_params['charge'].

        :param resname: (str)
        :param system: (openmm.System)
        :return: None
        """
        nonbonded_force = None
        for f in system.getForces():
            if f.getName() == "NonbondedForce":
                nonbonded_force = f
                break

        wat_params = {"charge":[], "sigma":[], "epsilon":[]}  # Store parameters in a dictionary
        for residue in self.topology.residues():
            if residue.name == resname:
                for atom in residue.atoms():
                    # Store the parameters of each atom
                    atom_params = nonbonded_force.getParticleParameters(atom.index)
                    wat_params["charge"].append( atom_params[0]) # has unit
                    wat_params["sigma"].append(  atom_params[1])  # has unit
                    wat_params["epsilon"].append(atom_params[2])  # has unit
                break  # Don't need to continue past the first instance
        self.wat_params = wat_params

    @staticmethod
    def _copy_nonbonded_setting_n2c(nonbonded_force: openmm.NonbondedForce, custom_nonbonded_force: openmm.CustomNonbondedForce):
        """
        Copy the nonbonded settings from NonbondedForce to CustomNonbondedForce.
        :param nonbonded_force:
        :param custom_nonbonded_force:
        :return: None
        """
        custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        custom_nonbonded_force.setCutoffDistance(nonbonded_force.getCutoffDistance())
        custom_nonbonded_force.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())
        custom_nonbonded_force.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
        custom_nonbonded_force.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())

    @staticmethod
    def _copy_nonbonded_setting_c2c(c1: openmm.CustomNonbondedForce, c2: openmm.CustomNonbondedForce):
        """
        Copy the nonbonded settings from CustomNonbondedForce to CustomNonbondedForce.
        :param c1:
        :param c2:
        :return: None
        """
        c2.setNonbondedMethod(c1.getNonbondedMethod())
        c2.setCutoffDistance(c1.getCutoffDistance())
        c2.setUseSwitchingFunction(c1.getUseSwitchingFunction())
        c2.setSwitchingDistance(c1.getSwitchingDistance())
        c2.setUseLongRangeCorrection(c1.getUseLongRangeCorrection())

    @staticmethod
    def _copy_exclusion_c2c(c1: openmm.CustomNonbondedForce, c2: openmm.CustomNonbondedForce):
        for exc_index in range(c1.getNumExclusions()):
            at1_index, at2_index = c1.getExclusionParticles(exc_index)
            c2.addExclusion(at1_index, at2_index)

    def customise_force_amber(self, system: openmm.System) -> None:
        """
        In Amber, NonbondedForce handles both electrostatics and vdW. This function will remove vdW from NonbondedForce
        and create a list of CustomNonbondedForce (in this case, only 1 in the list) to handle vdW, so that the
        interaction can be switched off for certain water.

        :param system: openmm.System
            The system to be converted.
        :return: None
        """
        self.system = system
        # check if the system is Amber
        if self.system_type != "Amber":
            raise ValueError("The system is not Amber. Please check the system.")

        self.logger.info("Try to customise a native Amber system")
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if force.getName() == "NonbondedForce":
                self.nonbonded_force = force

        energy_expression = ("U;"
                             "U = lambda_vdw * 4 * epsilon * x * (x - 1.0);"
                             "x = (1 / reff)^6;"
                             "reff = ((softcore_alpha * (1-lambda_vdw) + (r/sigma)^6))^(1/6);"
                             "lambda_vdw = is_real1 * is_real2 * lambda_switch;"
                             # is_s1, is_s2, is_r1, is_r2, switch_i, lambda_switch, soft-core
                             #   0.0,   0.0,   1.0,   1.0,      0.0,           1.0,       off
                             #   1.0,   0.0,   1.0,   1.0,      1.0, lambda_gc_vdw,        on
                             "lambda_switch = (1 - switch_indicator) + lambda_gc_vdw * switch_indicator;"
                             "switch_indicator = max(is_switching1, is_switching2);"
                             "epsilon = sqrt(epsilon1*epsilon2);"
                             "sigma = 0.5*(sigma1 + sigma2);")
        custom_nb_force = openmm.CustomNonbondedForce(energy_expression)
        # Add per particle parameters
        custom_nb_force.addPerParticleParameter("sigma")
        custom_nb_force.addPerParticleParameter("epsilon")
        custom_nb_force.addPerParticleParameter("is_real")
        custom_nb_force.addPerParticleParameter("is_switching")
        # Add global parameters
        custom_nb_force.addGlobalParameter('softcore_alpha', 0.5)
        custom_nb_force.addGlobalParameter('lambda_gc_vdw', 0.0) # lambda for vdw part of TI insertion/deletion
        # Transfer properties from the original force
        self._copy_nonbonded_setting_n2c(self.nonbonded_force, custom_nb_force)
        self.nonbonded_force.setUseDispersionCorrection(False)  # Turn off dispersion correction in NonbondedForce as it will only be used for Coulomb

        # remove vdw from NonbondedForce, and add particles to CustomNonbondedForce
        for at_index in range(self.nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            custom_nb_force.addParticle([sigma, epsilon, 1.0, 0.0]) # add
            self.nonbonded_force.setParticleParameters(at_index, charge, sigma, 0.0 * epsilon) # remove

        # Exceptions will not be changed in NonbondedForce, but will the corresponding pairs need to be excluded in CustomNonbondedForce
        for exception_idx in range(self.nonbonded_force.getNumExceptions()):
            [i, j, charge_product, sigma, epsilon] = self.nonbonded_force.getExceptionParameters(exception_idx)
            # Add vdw exclusion to CustomNonbondedForce
            custom_nb_force.addExclusion(i, j)

        self.system.addForce(custom_nb_force)
        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 0.0) # lambda for coulomb part of TI insertion/deletion

        # set is_switching to 1.0 for the switching water
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force(custom_nb_force)
        for at_index in self.water_res_2_atom[self.switching_water]:
            parameters = list(custom_nb_force.getParticleParameters(at_index))
            parameters[is_switching_index] = 1.0
            custom_nb_force.setParticleParameters(at_index, parameters)

        # set ParticleParameters and add ParticleParameterOffset for the switching water
        for at_index in self.water_res_2_atom[self.switching_water]:
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.addParticleParameterOffset("lambda_gc_coulomb", at_index, charge, 0.0, 0.0)
            self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon) # remove charge

        self.custom_nonbonded_force_list = [(is_real_index, is_switching_index, custom_nb_force)]

    def customise_force_charmm(self, system: openmm.System) -> None:
        """
        In Charmm, NonbondedForce handles electrostatics, and CustomNonbondedForce handles vdW. For vdW, this function will add
        perParticle parameters 'is_real', 'is_switching', global parameter ``lambda_gc_vdw`` to the CustomNonbondedForce.
        For Coulomb, this function will add ParticleParameterOffset to the switching water and ``lambda_gc_coulomb``
        to the NonbondedForce.

        The CustomNonbondedForce should have the following energy expression:
            '(a/r6)^2-b/r6; r6=r^6;a=acoef(type1, type2);b=bcoef(type1, type2)'
            or
            'acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6;'

        :param system: openmm.System
            The system to be converted.
        :return: None
        """
        self.system = system
        # check if the system is Charmm
        if self.system_type != "Charmm":
            raise ValueError("The system is not Charmm. Please check the system.")

        self.logger.info("Try to customise a native Charmm system")
        self.nonbonded_force = None
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if force.getName() == "NonbondedForce":
                self.nonbonded_force = force
            elif force.getName() == "CustomNonbondedForce":
                custom_nb_force = force
        if custom_nb_force.getNumGlobalParameters() != 0:
            raise ValueError("The CustomNonbondedForce should not have any global parameters. Please check the system.")

        energy_old = custom_nb_force.getEnergyFunction()
        if energy_old == '(a/r6)^2-b/r6; r6=r^6;a=acoef(type1, type2);b=bcoef(type1, type2)':
            energy_expression = (
                "U;"
                "U = lambda_vdw * ( (a/reff6)^2-b/reff6 );"
                "reff6 = sigma6 * (softcore_alpha * (1-lambda_vdw) + r^6/sigma6);" # reff^6
                "sigma6 = a^2 / b;"  # sigma^6
                "lambda_vdw = is_real1 * is_real2 * lambda_switch;"
                "lambda_switch = (1 - switch_indicator) + lambda_gc_vdw * switch_indicator;"
                "switch_indicator = max(is_switching1, is_switching2);"
                "a = acoef(type1, type2);"  # a = 2 * epsilon^0.5 * sigma^6
                "b = bcoef(type1, type2);"  # b = 4 * epsilon * sigma^6
            )
        elif energy_old == 'acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6;':
            energy_expression = (
                "U;"
                "U = lambda_vdw * ( (a/reff6)^2-b/reff6 );"
                "reff6 = sigma6 * (softcore_alpha * (1-lambda_vdw) + r^6/sigma6);" # reff^6
                "sigma6 = a / b;"  # sigma^6
                "lambda_vdw = is_real1 * is_real2 * lambda_switch;"
                "lambda_switch = (1 - switch_indicator) + lambda_gc_vdw * switch_indicator;"
                "switch_indicator = max(is_switching1, is_switching2);"
                "a = acoef(type1, type2);"  # a = 4 * epsilon * sigma^12
                "b = bcoef(type1, type2);"  # b = 4 * epsilon * sigma^6
            )
        else:
            raise ValueError(f"{energy_old} This energy expression in CustonNonbondedForce can not be converted. "
                             f"Currently, grandfep only supports the system that is prepared by CharmmPsfFile.CreateSystem() or ForceField.createSystem()")
        custom_nb_force.setEnergyFunction(energy_expression)
        # Add per particle parameters
        custom_nb_force.addPerParticleParameter("is_real")
        custom_nb_force.addPerParticleParameter("is_switching")
        for atom_idx in range(custom_nb_force.getNumParticles()):
            typ = custom_nb_force.getParticleParameters(atom_idx)
            custom_nb_force.setParticleParameters(atom_idx, [*typ, 1, 0])
        # Add global parameters
        custom_nb_force.addGlobalParameter('softcore_alpha', 0.5)
        custom_nb_force.addGlobalParameter('lambda_gc_vdw', 0.0) # lambda for vdw part of TI insertion/deletion
        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 0.0)  # lambda for coulomb part of TI insertion/deletion

        # set is_switching to 1.0 for the switching water
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force(custom_nb_force)
        for at_index in self.water_res_2_atom[self.switching_water]:
            parameters = list(custom_nb_force.getParticleParameters(at_index))
            parameters[is_switching_index] = 1.0
            custom_nb_force.setParticleParameters(at_index, parameters)
        self.custom_nonbonded_force_list = [(is_real_index, is_switching_index, custom_nb_force)]

        # set ParticleParameters and add ParticleParameterOffset for the switching water
        for at_index in self.water_res_2_atom[self.switching_water]:
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.addParticleParameterOffset("lambda_gc_coulomb", at_index, charge, 0.0, 0.0)
            self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon) # remove charge

    def customise_force_hybrid(self, system: openmm.System) -> None:
        """
        If the system is Hybrid, this function will add perParticleParameters ``is_real`` and ``is_switching``
        to the custom_nonbonded_force (openmm.openmm.CustomNonbondedForce) for vdw.

        +--------+------+------+------+------+------+------+
        | Groups | old  | core | new  | fix  | wat  | swit |
        +========+======+======+======+======+======+======+
        | old    |  C1  |      |      |      |      |      |
        +--------+------+------+------+------+------+------+
        | core   |  C1  |  C1  |      |      |      |      |
        +--------+------+------+------+------+------+------+
        | new    | None |  C1  |  C1  |      |      |      |
        +--------+------+------+------+------+------+------+
        | fix    |  C1  |  C1  |  C1  |  C4  |      |      |
        +--------+------+------+------+------+------+------+
        | wat    |  C1  |  C1  |  C1  |  C3  |  C3  |      |
        +--------+------+------+------+------+------+------+
        | swit   |  C1  |  C1  |  C1  |  C2  |  C2  |  C2  |
        +--------+------+------+------+------+------+------+


        Parameters
        ----------
        system :
            The system to be converted.

        Returns
        -------
        None
        """
        self.system = system
        # check if the system is Hybrid
        if self.system_type != "Hybrid":
            raise ValueError("The system is not Hybrid. Please check the system.")
        self.logger.info("Try to customise a native Hybrid system")
        self.nonbonded_force = None
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if force.getName() == "NonbondedForce":
                self.nonbonded_force = force
            elif force.getName() == "CustomNonbondedForce":
                custom_nb_force1 = force

        energy_old = custom_nb_force1.getEnergyFunction()
        if energy_old != 'U_sterics;U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;lambda_alpha = new_interaction*(1-lambda_sterics_insert) + old_interaction*lambda_sterics_delete;lambda_sterics = core_interaction*lambda_sterics_core + new_interaction*lambda_sterics_insert + old_interaction*lambda_sterics_delete;core_interaction = delta(unique_old1+unique_old2+unique_new1+unique_new2);new_interaction = max(unique_new1, unique_new2);old_interaction = max(unique_old1, unique_old2);epsilonA = sqrt(epsilonA1*epsilonA2);epsilonB = sqrt(epsilonB1*epsilonB2);sigmaA = 0.5*(sigmaA1 + sigmaA2);sigmaB = 0.5*(sigmaB1 + sigmaB2);':
            raise ValueError(
                f"{energy_old} This energy expression in CustonNonbondedForce can not be converted. "
                f"Currently, grandfep only supports the system that is prepared by HybridTopologyFactory with Beutler softcore.")

        energy_expression = (
            "U_sterics;"
            "U_sterics = 4*epsilon*x*(x-1.0);"
            "x = (1/reff_sterics)^6;"
            
            # 6. Calculate damped distance (reff/sigma)
            "reff_sterics = ((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);"
            
            # 5. Calculate the effective epsilon and sigma
            "epsilon = ((1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB) * scale_gc;"
            "sigma   = ((1-lambda_sterics)*sigmaA   + lambda_sterics*sigmaB  );"
            
            # 4. Soft-core on/off
            # new, old, swit, lambda_alpha
            #   0,   0,    0,                      0, SC off
            #   0,   1,    0,  lambda_sterics_delete, SC on
            #   1,   0,    0,1-lambda_sterics_insert, SC on
            #   1,   1,                                     won't happen, handled by exclusion
            #   0,   0,    1,        1-lambda_gc_vdw, SC on
            #   0,   1,    1, ???
            #   1,   0,    1, ???
            "lambda_alpha = new_interaction*(1-lambda_sterics_insert) + old_interaction*lambda_sterics_delete + switch_interaction*(1-lambda_gc_vdw);"
            
            # 3. Effective lambda. What is the overall progress of lambda evolving (0 -> 1)?
            # interaction, c, o, n,  lambda_sterics
            # core-core  , 1, 0, 0,  lambda_sterics_core
            #  old-old   , 0, 1, 0,  lambda_sterics_delete
            #  new-new   , 0, 0, 1,  lambda_sterics_insert
            #  old-new   , 0, 1, 1,  Exclusion
            #                 both_are_core                          one_is_new                              one_is_old
            "lambda_sterics = core_interaction*lambda_sterics_core + new_interaction*lambda_sterics_insert + old_interaction*lambda_sterics_delete;"
            
            # 2.3 Checking GC interactions. (real/ghost, switching/non-switching)
            # is_r1, is_r2, is_s1, is_s2, switch_inte,  lambda_pair,      scale_gc, comment
            #     1,     1,     0,     0,           0,            1,             1,   real-real
            #     1,     0,     0,     0,           0,            1,             0,   real-ghost
            #     0,     0,     0,     0,           0,            1,             0,  ghost-ghost
            #     1,     1,     1,     0,           1,lambda_gc_vdw, lambda_gc_vdw, switch-real
            "scale_gc = is_real1 * is_real2 * lambda_pair;"
            "lambda_pair = (1 - switch_interaction) + lambda_gc_vdw * switch_interaction;"
            "switch_interaction = max(is_switching1, is_switching2);"
            
            # 2.2 Checking core interactions. Is this a pair with 2 core atoms?
            "core_interaction = delta(unique_old1+unique_old2+unique_new1+unique_new2);"
            
            # 2.1 Checking unique interactions. Is this a pair of interaction that involves a unique atom?
            # There is exclusion between new-old, new_interaction and old_interaction will not be 1.0 simultaneously
            "new_interaction = max(unique_new1, unique_new2);"
            "old_interaction = max(unique_old1, unique_old2);"
            
            # 1. Applying combination rule. 
            "epsilonA = sqrt(epsilonA1*epsilonA2);"
            "epsilonB = sqrt(epsilonB1*epsilonB2);"
            "sigmaA = 0.5*(sigmaA1 + sigmaA2);"
            "sigmaB = 0.5*(sigmaB1 + sigmaB2);"
        )
        # Add per particle parameters
        custom_nb_force1.setEnergyFunction(energy_expression)
        custom_nb_force1.addPerParticleParameter("is_real")
        custom_nb_force1.addPerParticleParameter("is_switching")
        for atom_idx in range(custom_nb_force1.getNumParticles()):
            params = custom_nb_force1.getParticleParameters(atom_idx)
            custom_nb_force1.setParticleParameters(atom_idx, [*params, 1, 0]) # add is_real=1, is_switching=0

        # find env atoms in interaction groups in CustomNonbondedForce
        n_groups = custom_nb_force1.getNumInteractionGroups()
        if n_groups != 8:
            raise ValueError("The number of interaction groups is not 8")

        # Double check the interaction group
        env_index_list = [custom_nb_force1.getInteractionGroupParameters(i)[1] for i in [1, 3, 4]]
        flag_1 = env_index_list[0] == env_index_list[1]
        flag_2 = env_index_list[1] == env_index_list[2]
        if not flag_1 or not flag_2:
            raise ValueError(f"{env_index_list}\nThe selected interaction groups (1,3,4) are not the same.")
        # All water should be found in nv_index_list[0]
        water_at_list = []
        for res_index, at_list in self.water_res_2_atom.items():
            water_at_list.extend(at_list)
            for at_index in at_list:
                if at_index not in env_index_list[0]:
                    raise ValueError(f"water ({res_index=}) ({at_index=}) is not found in env group")
        water_group_set = set(water_at_list)
        fix_group_set = set(env_index_list[0]) - water_group_set
        # Add global parameters
        custom_nb_force1.addGlobalParameter('lambda_gc_vdw', 0.0)  # lambda for vdw part of TI insertion/deletion

        # set is_switching to 1.0 for the switching water
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force(custom_nb_force1)
        for at_index in self.water_res_2_atom[self.switching_water]:
            parameters = list(custom_nb_force1.getParticleParameters(at_index))
            parameters[is_switching_index] = 1.0
            custom_nb_force1.setParticleParameters(at_index, parameters)

        # C1 Done
        self.custom_nonbonded_force_list = [(is_real_index, is_switching_index, custom_nb_force1)]

        # Create C2 for switch-(fix, wat, switch)
        energy_expression = (
            "U_sterics;"
            "U_sterics = scale_gc*4*epsilon*x*(x-1.0);"
            "x = (1/reff_sterics)^6;"
            "reff_sterics = ((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);"
            "epsilon = sqrt(epsilon1*epsilon2);"
            "sigma = 0.5*(sigma1 + sigma2);"
            "lambda_alpha = switch_interaction*(1-lambda_gc_vdw);"
            "scale_gc = is_real1 * is_real2 * lambda_pair;"
            "lambda_pair = (1 - switch_interaction) + lambda_gc_vdw * switch_interaction;"
            "switch_interaction = max(is_switching1, is_switching2);"
        )
        custom_nb_force2 = openmm.CustomNonbondedForce(energy_expression)
        self._copy_nonbonded_setting_n2c(self.nonbonded_force, custom_nb_force2)
        custom_nb_force2.addPerParticleParameter("sigma")
        custom_nb_force2.addPerParticleParameter("epsilon")
        custom_nb_force2.addPerParticleParameter("is_real")
        custom_nb_force2.addPerParticleParameter("is_switching")
        custom_nb_force2.addGlobalParameter('softcore_alpha', 0.5)
        custom_nb_force2.addGlobalParameter('lambda_gc_vdw', 0.0)  # lambda for vdw part of TI insertion/deletion
        # Copy all the sigma, epsilon from NonbondedForce to C2
        for at in self.topology.atoms():
            at_index = at.index
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            custom_nb_force2.addParticle([sigma, epsilon, 1.0, 0.0])
        # Copy all the exclusion from C1 to C2
        self._copy_exclusion_c2c(custom_nb_force1, custom_nb_force2)
        # Add switch-(fix, wat, switch) interaction group
        # if H has no vdw, remove them from water_group_set
        water_group = []
        switch_group = []
        for res_index, at_list in self.water_res_2_atom.items():
            if res_index == self.switching_water:
                for at_index in at_list:
                    charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                    if epsilon > 1e-8 * unit.kilojoule_per_mole:
                        switch_group.append(at_index)
            else:
                for at_index in at_list:
                    charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                    if epsilon > 1e-8 * unit.kilojoule_per_mole:
                        water_group.append(at_index)
        water_group_set = set(water_group)
        switch_group_set = set(switch_group)


        custom_nb_force2.addInteractionGroup(switch_group_set, fix_group_set | water_group_set)
        custom_nb_force2.addInteractionGroup(switch_group_set, switch_group_set)

        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force(custom_nb_force2)
        for at_index in self.water_res_2_atom[self.switching_water]:
            parameters = list(custom_nb_force2.getParticleParameters(at_index))
            parameters[is_switching_index] = 1.0
            custom_nb_force2.setParticleParameters(at_index, parameters)

        # C2 Done
        self.custom_nonbonded_force_list.append(
            (is_real_index, is_switching_index, custom_nb_force2)
        )
        self.system.addForce(custom_nb_force2)

        # Create C3 for wat-fix, wat-wat
        energy_expression = (
            "U_sterics;"
            "U_sterics = scale_gc*4*epsilon*x*(x-1.0);"
            "x = (sigma/r)^6;"
            "epsilon = sqrt(epsilon1*epsilon2);"
            "sigma = 0.5*(sigma1 + sigma2);"
            "scale_gc = is_real1 * is_real2;"
        )
        custom_nb_force3 = openmm.CustomNonbondedForce(energy_expression)
        self._copy_nonbonded_setting_n2c(self.nonbonded_force, custom_nb_force3)
        custom_nb_force3.addPerParticleParameter("sigma")
        custom_nb_force3.addPerParticleParameter("epsilon")
        custom_nb_force3.addPerParticleParameter("is_real")
        # Copy all the sigma, epsilon from NonbondedForce to C3
        for at in self.topology.atoms():
            at_index = at.index
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            custom_nb_force3.addParticle([sigma, epsilon, 1.0])
        # Copy all the exclusion from C1 to C3
        self._copy_exclusion_c2c(custom_nb_force1, custom_nb_force3)
        # Add water-water, water-fix interaction group
        custom_nb_force3.addInteractionGroup(water_group_set, fix_group_set)
        custom_nb_force3.addInteractionGroup(water_group_set, water_group_set)
        # C3 Done
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force(custom_nb_force3)
        self.custom_nonbonded_force_list.append(
            (is_real_index, is_switching_index, custom_nb_force3)
        )
        self.system.addForce(custom_nb_force3)



        # Create C4 for fix-fix interaction
        energy_expression = (
            "U;"
            "U = 4*epsilon*x*(x-1.0);"
            "x = (sigma/r)^6;"
            "epsilon = sqrt(epsilon1*epsilon2);"
            "sigma = 0.5*(sigma1 + sigma2);"
        )
        custom_nb_force4 = openmm.CustomNonbondedForce(energy_expression)
        self._copy_nonbonded_setting_n2c(self.nonbonded_force, custom_nb_force4)
        custom_nb_force4.addPerParticleParameter("sigma")
        custom_nb_force4.addPerParticleParameter("epsilon")
        # Copy all the sigma, epsilon from NonbondedForce to C4
        for at in self.topology.atoms():
            at_index = at.index
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            custom_nb_force4.addParticle([sigma, epsilon])
        # Copy all the exclusion from C1 to C4
        self._copy_exclusion_c2c(custom_nb_force1, custom_nb_force4)
        # Add fix-fix interaction group
        custom_nb_force4.addInteractionGroup(fix_group_set, fix_group_set)
        self.system.addForce(custom_nb_force4)
        # C4 no longer need updateParametersInContext anymore. It will not be appended to self.custom_nonbonded_force_list


        # Remove a vdw from nonbonded_force
        for at in self.topology.atoms():
            at_index = at.index
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.setParticleParameters(at_index, charge, sigma, 0.0*epsilon)

        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 0.0)  # lambda for coulomb part of TI insertion/deletion
        # set ParticleParameters and add ParticleParameterOffset for the switching water
        for at_index in self.water_res_2_atom[self.switching_water]:
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.addParticleParameterOffset("lambda_gc_coulomb", at_index, charge, 0.0, 0.0)
            self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon) # remove charge

    def _turn_off_vdw(self):
        """
        Turn off all vdw in self.custom_nonbonded_force_list.

        This function is only used for debugging purpose.

        :return: None
        """
        # All epsilon to 0.0
        for _r, _s, custom_nb_force in self.custom_nonbonded_force_list:
            for atom in self.topology.atoms():
                at_ind = atom.index
                sigmaA, epsilonA, sigmaB, epsilonB, unique_old, unique_new, is_real, is_switching = custom_nb_force.getParticleParameters(at_ind)
                custom_nb_force.setParticleParameters(at_ind, [sigmaA, 0.0*epsilonA, sigmaB, 0.0*epsilonB, unique_old, unique_new, is_real*0.0, is_switching])
            custom_nb_force.updateParametersInContext(self.simulation.context)

    def set_ghost_list(self, ghost_list: list, check_system: bool = True) -> None:
        """
        Update the water residues in `ghost_list` to ghost.

        If `check_system` is True, `self.check_ghost_list` will be called to validate the consistency of self.ghost_list
        with `self.custom_nonbonded_force_list` and `self.nonbonded_force`. If the validation fails,
        a ValueError will be raised.

        Parameters
        ----------
        ghost_list : list
            A list of residue indices (integers) that should be marked as ghost waters.

        check_system : bool, optional
            If True, perform validation to ensure the internal force parameters are consistent with
            the updated `ghost_list`. Default is True.

        Returns
        -------
        None
        """
        if self.switching_water in ghost_list:
            raise ValueError("Switching water should never be set to ghost.")

        # set self.custom_nonbonded_force_list for vdw
        for is_real_index, is_switching_index, c_nbforce in self.custom_nonbonded_force_list:
            for res_index in self.ghost_list:
                if res_index not in self.water_res_2_atom:
                    raise ValueError(f"The residue {res_index} is water.")
            for res_index, at_list in self.water_res_2_atom.items():
                if res_index == self.switching_water:
                    continue
                for at_index in at_list:
                    parameters = list(c_nbforce.getParticleParameters(at_index))
                    parameters[is_real_index] = 1.0
                    c_nbforce.setParticleParameters(at_index, parameters)

            for res_index in ghost_list:
                if res_index not in self.water_res_2_atom:
                    raise ValueError(f"The residue {res_index} is not water.")
                for at_index in self.water_res_2_atom[res_index]:
                    parameters = list(c_nbforce.getParticleParameters(at_index))
                    parameters[is_real_index] = 0.0
                    c_nbforce.setParticleParameters(at_index, parameters)

        # set self.nonbonded_force for coulomb
        for res_index, at_list in self.water_res_2_atom.items():
            if res_index == self.switching_water:
                continue
            for at_index, wat_chg in zip(self.water_res_2_atom[res_index], self.wat_params['charge']):
                charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                self.nonbonded_force.setParticleParameters(at_index, wat_chg, sigma, epsilon)

        for res_index in ghost_list:
            for at_index in self.water_res_2_atom[res_index]:
                charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon)

        # update context
        self.nonbonded_force.updateParametersInContext(self.simulation.context)
        for _r, _s, custom_nb_force in self.custom_nonbonded_force_list:
            custom_nb_force.updateParametersInContext(self.simulation.context)
        self.ghost_list = ghost_list

        if check_system:
            self.check_ghost_list()

    def get_ghost_list(self, check_system: bool = False) -> list:
        """
        Get a copy of the current ghost water list.

        If `check_system` is True, `self.check_ghost_list` will be called to validate the consistency of self.ghost_list
        with `self.custom_nonbonded_force` and `self.nonbonded_force`. If the validation fails,
        a ValueError will be raised.

        Parameters
        ----------
        check_system :
            If True, self.check_ghost_list() will be called to perform consistency checks in the system forces
            to ensure that the ghost list is correct.
            Default is False.

        Returns
        -------
        list
            A copy of `self.ghost_list`, which contains the residue indices of ghost waters.
        """
        if check_system:
            self.check_ghost_list()
        return deepcopy(self.ghost_list)

    def check_ghost_list(self):
        """
        Loop over all water particles in the system to validate that `self.ghost_list` correctly reflects
        the current ghost and switching water configuration.

        - Ghost water:
            1. `is_real = 0.0` and `is_switching = 0.0` in every CustomNonbondedForce in `self.custom_nonbonded_force_list`
            2. Charge = 0.0 in `self.nonbonded_force`

        - Real and not switching water:
            1. `is_real = 1.0` in every CustomNonbondedForce in `self.custom_nonbonded_force_list`
            2. Charge = Proper_water_charge in `self.nonbonded_force`

        - Switching water:
            1. `is_real = 1.0` in every CustomNonbondedForce in `self.custom_nonbonded_force_list`
            2. Charge = 0.0 in `self.nonbonded_force`

        The switching water should not be present in `self.ghost_list`.

        Raises
        ------
        ValueError
            If any condition for ghost, real, or switching water is violated.

        Returns
        -------
        None
        """
        if self.switching_water in self.ghost_list:
            raise ValueError("Switching water should never be set to ghost.")

        # check vdW in self.custom_nonbonded_force
        for is_real_index, is_switching_index, custom_nb_force in self.custom_nonbonded_force_list:
            for res, at_index_list in self.water_res_2_atom.items():
                if res in self.ghost_list:
                    # ghost water should have is_real = 0.0, is_switching = 0.0
                    for at_index in at_index_list:
                        parameters = custom_nb_force.getParticleParameters(at_index)
                        if parameters[is_real_index] > 1e-8:
                            raise ValueError(
                                f"The water {res} (real) atom {at_index} has is_real = {parameters[is_real_index]}.")
                        if is_switching_index is not None and parameters[is_switching_index] > 1e-8:
                            raise ValueError(
                                f"The water {res} (real) atom {at_index} has is_switching = {parameters[is_switching_index]}.")
                else:
                    # real water should have is_real = 1.0
                    for at_index in at_index_list:
                        parameters = custom_nb_force.getParticleParameters(at_index)
                        if parameters[is_real_index] < 0.99999999:
                            raise ValueError(
                                f"The water {res} at {at_index} has is_real = {parameters[is_real_index]}."
                            )

        # check coulomb in self.nonbonded_force
        for res, at_index_list in self.water_res_2_atom.items():
            if res in self.ghost_list:
                # ghost water should have charge = 0.0
                for at_index in at_index_list:
                    charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                    if abs(charge.value_in_unit(unit.elementary_charge)) > 1e-8:
                        raise ValueError(
                            f"The water {res} at {at_index} has charge = {charge}."
                        )
            else:
                if res == self.switching_water:
                    # switching water should have charge = 0.0
                    for at_index in at_index_list:
                        charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                        if abs(charge.value_in_unit(unit.elementary_charge)) > 1e-8:
                            raise ValueError(
                                f"The water {res} (switching) at {at_index} has charge = {charge}. It should be 0.0."
                            )
                else:
                    # real water should have proper charge
                    for at_index, wat_chg in zip(self.water_res_2_atom[res], self.wat_params['charge']):
                        charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                        if not np.allclose(charge.value_in_unit(unit.elementary_charge), wat_chg.value_in_unit(unit.elementary_charge)):
                            raise ValueError(
                                f"The water {res} at {at_index} has charge = {charge}, should be {wat_chg.value_in_unit(unit.elementary_charge)}."
                            )

    def check_switching(self):
        """
        Loop over all water particles in the system to make sure the one water given by self.switching_water is the
        only one that is switching.

        - Switching water:
            1. `is_real = 1.0` and `is_switching = 1.0` in `self.custom_nonbonded_force`
            2. Charge = 0.0 in `self.nonbonded_force` (ParticleParameters)
            3. `chargeScale = proper_water_charge` in `ParticleParameterOffsets`

        - Non-switching waters:
            1. `is_switching = 0.0` in `self.custom_nonbonded_force`
            2. No `ParticleParameterOffsets` in `self.nonbonded_force`

        Raises
        ------
        ValueError
            If any switching or non-switching water fails to meet the expected criteria.

        Returns
        -------
        None
        """
        # check vdW in self.custom_nonbonded_force
        for is_real_index, is_switching_index, custom_nb_force in self.custom_nonbonded_force_list:
            for res, at_index_list in self.water_res_2_atom.items():
                if res != self.switching_water:
                    # check is_switching should be 0.0
                    for at_index in at_index_list:
                        parameters = custom_nb_force.getParticleParameters(at_index)
                        if is_switching_index is not None and parameters[is_switching_index] > 1e-8:
                            self.logger.warning(
                                f"The water {res} at {at_index} has is_switching = {parameters[is_switching_index]}."
                            )
                else:
                    # check is_switching should be 1.0, is_real should be 1.0
                    for at_index in at_index_list:
                        parameters = custom_nb_force.getParticleParameters(at_index)
                        if is_switching_index is not None and parameters[is_switching_index] < 0.99999999:
                            raise ValueError(
                                f"The water {res} at {at_index} has is_switching = {parameters[is_switching_index]}."
                            )
                        if parameters[is_real_index] < 0.99999999:
                            raise ValueError(
                                f"The water {res} at {at_index} has is_real = {parameters[is_real_index]}."
                            )

        # check coulomb in self.nonbonded_force
        particle_parameter_offset_dict = self.get_particle_parameter_offset_dict()

        for res, at_index_list in self.water_res_2_atom.items():
            if res != self.switching_water:
                # non-switching water
                for at_index in at_index_list:
                    # no ParticleParameterOffsets
                    if at_index in particle_parameter_offset_dict:
                        raise ValueError(f"Water {res} (no switching) at {at_index} has a ParticleParameterOffset {particle_parameter_offset_dict[at_index]}.")
            else:
                # switching water
                for at_index, wat_chg in zip(self.water_res_2_atom[res], self.wat_params['charge']):
                    # check ParticleParameterOffsets
                    if at_index in particle_parameter_offset_dict:
                        param_offset_index, param_name, _at_index, charge_scale, sigma_scale, epsilon_scale = particle_parameter_offset_dict[at_index]
                        if param_name != "lambda_gc_coulomb":
                            raise ValueError(f"Water {res} (is switching) at {at_index} has a nonbonded force parameter offset {param_name}. ")
                        if np.allclose(charge_scale, wat_chg.value_in_unit(unit.elementary_charge)) is False:
                            raise ValueError(f"Water {res} (is switching) at {at_index} has wrong ParticleParameterOffset for chargeScale {particle_parameter_offset_dict[at_index]} .")
                        if sigma_scale > 1e-8 or epsilon_scale > 1e-8:
                            raise ValueError(f"Water {res} (is switching) at {at_index} has non-zero ParticleParameterOffset for vdw {particle_parameter_offset_dict[at_index]} .")
                    else:
                        raise ValueError(f"Water {res} (is switching) at {at_index} does not have a ParticleParameterOffset.")

                    # check charge in ParticleParameters
                    charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                    if not np.allclose(charge.value_in_unit(unit.elementary_charge), 0.0):
                        raise ValueError(f"Water {res} (is switching) at {at_index} has ParticleParameters of charge = {charge}, should be 0.0.")

    @staticmethod
    def get_particle_parameter_index_cust_nb_force(custom_nonbonded_force: openmm.CustomNonbondedForce) -> tuple[int, int]:
        """
        Get the indices of `is_real` and `is_switching` parameters in the perParticleParameters
        of `self.custom_nonbonded_force`.

        Parameters
        ----------
        custom_nonbonded_force :
            The CustomNonbondedForce to be checked.

        Returns
        -------
        tuple of int
            A tuple `(is_real_index, is_switching_index)` indicating the indices of the corresponding
            parameters in `self.custom_nonbonded_force.getPerParticleParameterName(i)`.
        """
        # make sure we have the correct index for is_real and is_switching in perParticleParameters
        is_real_index = None
        is_switching_index = None
        for i in range(custom_nonbonded_force.getNumPerParticleParameters()):
            name = custom_nonbonded_force.getPerParticleParameterName(i)
            if name == "is_real":
                is_real_index = i
            elif name == "is_switching":
                is_switching_index = i
        return is_real_index, is_switching_index

    def get_particle_parameter_offset_dict(self) -> dict:
        """
        Retrieve all `ParticleParameterOffset` entries from `self.nonbonded_force`.

        Returns
        -------
        dict
            A dictionary mapping atom indices to their corresponding `ParticleParameterOffset` data.

            The key is the atom index. Each value is a list with the following structure:
            [
            param_offset_index,
            global_parameter_name,
            atom_index,
            chargeScale,
            sigmaScale,
            epsilonScale
            ]
        """
        particle_parameter_offset_dict = {}
        for param_offset_index in range(self.nonbonded_force.getNumParticleParameterOffsets()):
            offset = self.nonbonded_force.getParticleParameterOffset(param_offset_index)
            at_index = offset[1]
            particle_parameter_offset_dict[at_index] = [param_offset_index, *offset]
        return particle_parameter_offset_dict

class _ReplicaExchangeMixin:
    """
    This class is a mixin for the MPI sampler class. It contains methods that are used in replica exchange
    """
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

        self.lambda_state_index = init_lambda_state
        self.lambda_dict = lambda_dict
        # all gather the lambda states
        self.lambda_states_list = self.comm.allgather(self.lambda_state_index)
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
        self.logger.info(f"init_lambda_state={self.lambda_state_index}")
        for lam, val_list in self.lambda_dict.items():
            self.logger.info(f"Set {lam}={val_list[self.lambda_state_index]}")
            self.simulation.context.setParameter(lam, val_list[self.lambda_state_index])

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

        # re_step has to be the same across all replicas. If not, raise an error.
        re_step_all = self.comm.allgather(self.re_step)
        if len(set(re_step_all)) != 1:
            raise ValueError(f"RE step is not the same across all replicas: {re_step_all}")

    def set_lambda_state(self, i: int) -> None:
        """
        Set the lambda state index for this replica.

        Parameters
        ----------
        i : int
            The lambda state index for this replica, counting from 0

        Returns
        -------
        None
        """
        if i >= self.n_lambda_states:
            raise ValueError(f"The lambda state index {i} is larger than the number of lambda states {self.n_lambda_states}")
        self.lambda_state_index = i
        for lam, val_list in self.lambda_dict.items():
            self.simulation.context.setParameter(lam, val_list[self.lambda_state_index])

    def _calc_neighbor_reduced_energy(self, set_lam_back: bool=True) -> np.array:
        """
        Use the current configuration and compute the energy of the neighboring sampling state. Later BAR can be performed on this data.

        Returns
        -------
        reduced_energy :
            reduced energy in kBT, no unit
        set_lam_back :
            If True, set the lambda state back to the original state after calculation.
            Default is True.
        """
        lambda_state_index_old = self.lambda_state_index
        reduced_energy = np.zeros(self.n_lambda_states, dtype=np.float64)
        state = self.simulation.context.getState(getEnergy=True)
        reduced_energy[lambda_state_index_old] = state.getPotentialEnergy() / self.kBT

        # when there is left neighbor
        if lambda_state_index_old >=1:
            i = lambda_state_index_old - 1
            self.set_lambda_state(i)
            state = self.simulation.context.getState(getEnergy=True)
            reduced_energy[i] = state.getPotentialEnergy() / self.kBT

        # when there is right neighbor
        if lambda_state_index_old < self.n_lambda_states-1:
            i = lambda_state_index_old + 1
            self.set_lambda_state(i)
            state = self.simulation.context.getState(getEnergy=True)
            reduced_energy[i] = state.getPotentialEnergy() / self.kBT

        # back to the original state
        if set_lam_back:
            self.set_lambda_state(lambda_state_index_old)
        return reduced_energy

    def _calc_full_reduced_energy(self, set_lam_back: bool=True) -> np.array:
        """
        Use the current configuration and compute the energy of all the sampling states. Later MBAR can be performed on this data.

        Returns
        -------
        reduced_energy :
            reduced energy in kBT, no unit
        set_lam_back :
            If True, set the lambda state back to the original state after calculation.
            Default is True.
        """
        lambda_state_index_old = self.lambda_state_index
        reduced_energy = np.zeros(self.n_lambda_states, dtype=np.float64)
        state = self.simulation.context.getState(getEnergy=True)
        reduced_energy[lambda_state_index_old] = state.getPotentialEnergy() / self.kBT

        for i in range(self.n_lambda_states):
            if i != lambda_state_index_old:
                # set global parameters
                self.set_lambda_state(i)
                state = self.simulation.context.getState(getEnergy=True)
                reduced_energy[i] = state.getPotentialEnergy() / self.kBT

        # back to the original state
        if set_lam_back:
            self.set_lambda_state(lambda_state_index_old)
        return reduced_energy

    def replica_exchange_global_param(self, calc_neighbor_only: bool = True):
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
            The key is the MPI rank pairs, and the value is the decision and the acceptance ratio.

        """
        # re_step has to be the same across all replicas. If not, raise an error.
        re_step_all = self.comm.allgather(self.re_step)
        if len(set(re_step_all)) != 1:
            raise ValueError(f"RE step is not the same across all replicas: {re_step_all}")

        self.logger.info(f"RE Step {self.re_step}")

        lambda_state_index_old = self.lambda_state_index
        self.lambda_states_list = self.comm.gather(self.lambda_state_index, root=0) # map from rank to lambda_state_index

        # calculate reduced energy
        if calc_neighbor_only:
            reduced_energy = self._calc_neighbor_reduced_energy(False)
        else:
            reduced_energy = self._calc_full_reduced_energy(False)  # kBT unit

        reduced_energy_all = self.comm.gather(reduced_energy, root=0)

        # rank 0 log the reduced energy matrix in order
        re_decision = None  # placeholder on other ranks
        if self.rank == 0:

            r_energy_list = [[l_index, re_e] for l_index, re_e in zip(self.lambda_states_list, reduced_energy_all)]
            r_energy_list_order = sorted(r_energy_list, key=lambda x: x[0])
            for l_index, re_e in r_energy_list_order:
                e_str = ",".join([f"{i:14f}" for i in re_e])
                self.logger.info(f"{l_index}: " + e_str)


            reduced_energy_matrix = np.array(reduced_energy_all)
            # rank 0 decides the exchange

            re_decision = {}
            re_dec_state = {}
            lambda_states_list_order = sorted(self.lambda_states_list)
            map_l_index_2_rank = {l_index: rank for rank, l_index in enumerate(self.lambda_states_list)}
            for rep_i in range(self.re_step % 2, self.size - 1, 2):
                state_i = lambda_states_list_order[rep_i]
                state_j = lambda_states_list_order[rep_i+1]
                rank_i = map_l_index_2_rank[state_i]
                rank_j = map_l_index_2_rank[state_j]

                delta_energy = (reduced_energy_matrix[  rank_i, state_j] + reduced_energy_matrix[rank_j, state_i]
                                - reduced_energy_matrix[rank_i, state_i] - reduced_energy_matrix[rank_j, state_j])
                accept_prob = math.exp(-delta_energy)
                acc = min(1.0, accept_prob)
                if np.random.rand() < accept_prob:
                    re_decision[rank_i] = (True, state_j)
                    re_decision[rank_j] = (True, state_i)
                    re_dec_state[(min(state_i, state_j), max(state_i, state_j))] = (True,  acc)
                else:
                    re_decision[rank_i] = (False, state_i)
                    re_decision[rank_j] = (False, state_j)
                    re_dec_state[(min(state_i, state_j), max(state_i, state_j))] = (False, acc)
            # log re_decision
            msg_ex = "Repl ex :"
            msg_pr = "Repl pr :"
            x_dict = {True: "x", False: " "}
            if self.re_step % 2 == 1:
                msg_ex += f"{lambda_states_list_order[0]:2}   "
                msg_pr +=  "     "
            state_pair_list = sorted(re_dec_state.keys())
            msg_ex += "   ".join([f"{i[0]:2} {x_dict[re_dec_state[i][0]]} {i[1]:2}" for i in state_pair_list])
            msg_pr += "   ".join([f"{re_dec_state[i][1]:7.3f}" for i in state_pair_list])
            if not (self.size-2, self.size-1) in re_dec_state:
                msg_ex += f"   {lambda_states_list_order[self.size - 1]:2}"
            self.logger.info(msg_ex)
            self.logger.info(msg_pr)

        re_decision = self.comm.bcast(re_decision, root=0)

        # set state for each rank
        exchange = False
        new_state = lambda_state_index_old
        if self.rank in re_decision:
            exchange, new_state_tmp = re_decision[self.rank]
            if exchange:
                new_state = new_state_tmp
        self.set_lambda_state(new_state)

        self.re_step += 1
        return re_decision, exchange



