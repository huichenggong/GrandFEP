from copy import deepcopy
import logging
from typing import Union
from pathlib import Path

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
    per-particle parameters `is_real` and `is_switching` interact with the global parameter `lambda_gc_vdw`.

    Electrostatic interactions are handled by `self.nonbonded_force` (`openmm.NonbondedForce`). Ghost waters
    are given zero charge, and switching waters use `ParticleParameterOffset` with a global parameter
    `lambda_gc_coulomb`.

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


    Attributes
    ----------
    logger : logging.Logger

    compound_integrator : openmm.CompoundIntegrator
        2 integrators in this attribute. The 1st one is for Canonical simulation, the 2nd one is for non-equilibirum insertion/deletion.

    simulation : app.Simulation
        simulation ties together Topology, System, Integrator, and Context in this sampler.

    temperature : unit.Quantity
        The reference temperature for the system, with proper units (e.g., kelvin).

    kBT : unit.Quantity
        k_B * T, with unit.

    topology : app.Topology
        The OpenMM Topology object. All the res_name, atom_index, atom_name, etc. are in this topology.

    Adam_box : unit.Quantity
        Adam value, considering the whole simulation box and all water molecules.

    Adam_GCMC : unit.Quantity
        Adam value, considering the selected sphere and water molecules in the sphere.

    ghost_list : list
        A list of residue indices of water that are set to ghost. Should only be modified with set_ghost_list(),
        and get by get_ghost_list().

    water_res_2_atom : dict
        A dictionary of residue index to a list of atom indices of water.

    water_res_2_O : dict
        A dictionary of residue index to a list of atom indices of water oxygen.

    switching_water : int
        The residue index of the switching water. The switching water will be set as the last water during initialization.
        It should not be changed during the simulation, as the ParticleParameterOffset can not be updated in NonbondedForce.

    num_of_points_water : int
        The number of points in the water model. 3 for TIP3P, 4 for OPC.

    custom_nonbonded_force : openmm.CustomNonbondedForce
        This force handles vdW. It has PerParticleParameter is_real and is_switching to control real/ghost and
        switching/non-switching. It also has a global parameter **lambda_gc_vdw** to control the switching water.

    nonbonded_force : openmm.NonbondedForce
        This force handles Coulomb. The switching water has ParticleParameterOffset with global parameter
        **lambda_gc_coulomb** to control the switching water.

    wat_params : dict
        A dictionary to track the nonbonded parameter of water. The keys are "charge", "sigma", "epsilon",
        The values are a list of parameters with unit.

    system_type : str
        The type of the system. Can be Amber, Charmm or Hybrid depending on the system and energy expression in the given CustomNonbondedForce.

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
                 ):
        """
        Initialize the BaseGrandCanonicalMonteCarloSampler.
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

        # prepare logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                                    "%m-%d %H:%M:%S"))
        self.logger.addHandler(file_handler)
        self.logger.info("Initializing BaseGrandCanonicalMonteCarloSampler")

        self.system = None
        self.compound_integrator = None
        self.simulation = None
        self.topology = topology

        # constants and simulation configuration
        self.kBT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.temperature = temperature
        self.Adam_box = None
        self.Adam_GCMC = None


        self.ghost_list = []
        self.water_res_2_atom = None
        self.water_res_2_O = None
        self.switching_water = None
        self.num_of_points_water = None

        # system and force field related attributes
        self.custom_nonbonded_force = None
        self.nonbonded_force = None
        self.wat_params = None
        self.system_type = None

        # preparation based on the topology
        self.logger.info("Check topology")
        self.num_of_points_water = self._check_water_points(water_resname)
        self._find_all_water(water_resname, water_O_name)

        # preparation based on the system and force field
        self.logger.info("Prepare system")
        self.system_type = self._check_system(system)
        self._get_water_parameters(water_resname, system)

        if self.system_type == "Amber":
            self._customise_force_amber(system)
        elif self.system_type == "Charmm":
            self._customise_force_charmm(system)
        elif self.system_type == "Hybrid":
            self._customise_force_hybrid(system)
        else:
            raise ValueError(f"The system ({self.system_type}) cannot be customized. Please check the system.")

        # preparation of integrator, simulation
        self.logger.info("Prepare integrator and simulation")
        self.compound_integrator = openmm.CompoundIntegrator()
        integrator = BAOABIntegrator(temperature, collision_rate, timestep)
        self.compound_integrator.addIntegrator(integrator) # for EQ run
        integrator = BAOABIntegrator(temperature, collision_rate, timestep)
        self.compound_integrator.addIntegrator(integrator) # for NCMC run

        self.simulation = app.Simulation(self.topology, self.system, self.compound_integrator, platform)



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
                self.water_res_2_O[res.index] = []
                for atom in res.atoms():
                    self.water_res_2_atom[res.index].append(atom.index)
                    if atom.name == water_O_name:
                        self.water_res_2_O[res.index].append(atom.index)
        self.logger.info(f"Water {self.switching_water} will be set as the switching water")

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
            CustomNonbondedForce is '(a/r6)^2-b/r6; r6=r^6;a=acoef(type1, type2);b=bcoef(type1, type2)' or
            'acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6;'

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

        wat_params = {"charge":[]}  # Store parameters in a dictionary
        for residue in self.topology.residues():
            if residue.name == resname:
                for atom in residue.atoms():
                    # Store the parameters of each atom
                    atom_params = nonbonded_force.getParticleParameters(atom.index)
                    wat_params["charge"].append(atom_params[0]) # has unit
                break  # Don't need to continue past the first instance
        self.wat_params = wat_params

    def _customise_force_amber(self, system):
        """
        In Amber,  NonbondedForce handles both electrostatics and vdW. This function will remove vdW from NonbondedForce
        and create a CustomNonbondedForce to handle vdW, so that the interaction can be switched off for certain water.

        :param system: openmm.System
            The system to be converted.
        :return: None
        """
        self.system = deepcopy(system)
        # check if the system is Amber
        if self.system_type != "Amber":
            raise ValueError("The system is not Amber. Please check the system.")

        self.logger.info("Try to customise a native Amber system")
        self.custom_nonbonded_force = None
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if force.getName() == "NonbondedForce":
                self.nonbonded_force = force

        energy_expression = ("U;"
                             "U = lambda_vdw * 4 * epsilon * x * (x - 1.0);"
                             "x = (sigma / reff)^6;"
                             "reff = sigma * ((softcore_alpha * (1-lambda_vdw) + (r/sigma)^6))^(1/6);"
                             "lambda_vdw = is_real1 * is_real2 * lambda_switch;"
                             "lambda_switch = (1 - switch_indicator) + lambda_gc_vdw * switch_indicator;"
                             "switch_indicator = max(is_switching1, is_switching2);"
                             "epsilon = sqrt(epsilon1*epsilon2);"
                             "sigma = 0.5*(sigma1 + sigma2);")
        self.custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
        # Add per particle parameters
        self.custom_nonbonded_force.addPerParticleParameter("sigma")
        self.custom_nonbonded_force.addPerParticleParameter("epsilon")
        self.custom_nonbonded_force.addPerParticleParameter("is_real")
        self.custom_nonbonded_force.addPerParticleParameter("is_switching")
        # Add global parameters
        self.custom_nonbonded_force.addGlobalParameter('softcore_alpha', 0.5)
        self.custom_nonbonded_force.addGlobalParameter('lambda_gc_vdw', 0.0) # lambda for vdw part of TI insertion/deletion
        # Transfer properties from the original force
        self.custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        self.custom_nonbonded_force.setUseSwitchingFunction(self.nonbonded_force.getUseSwitchingFunction())
        self.custom_nonbonded_force.setCutoffDistance(self.nonbonded_force.getCutoffDistance())
        self.custom_nonbonded_force.setSwitchingDistance(self.nonbonded_force.getSwitchingDistance())
        self.custom_nonbonded_force.setUseLongRangeCorrection(self.nonbonded_force.getUseDispersionCorrection())
        self.nonbonded_force.setUseDispersionCorrection(False)  # Turn off dispersion correction in NonbondedForce as it will only be used for Coulomb

        # remove vdw from NonbondedForce, and add particles to CustomNonbondedForce
        for at_index in range(self.nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.custom_nonbonded_force.addParticle([sigma, epsilon, 1.0, 0.0]) # add
            self.nonbonded_force.setParticleParameters(at_index, charge, sigma, 0.0 * epsilon) # remove

        # Exceptions will not be changed in NonbondedForce, but will the corresponding pairs need to be excluded in CustomNonbondedForce
        for exception_idx in range(self.nonbonded_force.getNumExceptions()):
            [i, j, charge_product, sigma, epsilon] = self.nonbonded_force.getExceptionParameters(exception_idx)
            # Add vdw exclusion to CustomNonbondedForce
            self.custom_nonbonded_force.addExclusion(i, j)

        self.system.addForce(self.custom_nonbonded_force)
        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 0.0) # lambda for coulomb part of TI insertion/deletion

        # set is_switching to 1.0 for the switching water
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force()
        for at_index in self.water_res_2_atom[self.switching_water]:
            parameters = list(self.custom_nonbonded_force.getParticleParameters(at_index))
            parameters[is_switching_index] = 1.0
            self.custom_nonbonded_force.setParticleParameters(at_index, parameters)

        # set ParticleParameters and add ParticleParameterOffset for the switching water
        for at_index in self.water_res_2_atom[self.switching_water]:
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.addParticleParameterOffset("lambda_gc_coulomb", at_index, charge, 0.0, 0.0)
            self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon) # remove charge

        # add Derivative
        # self.custom_nonbonded_force.addEnergyParameterDerivative('lambda_gc_vdw')

    def _customise_force_charmm(self, system):
        """
        In Charmm, NonbondedForce handles electrostatics, and CustomNonbondedForce handles vdW. For vdW, this function will add
        perParticle parameters 'is_real', 'is_switching', global parameter 'lambda_gc_vdw' to the CustomNonbondedForce.
        For Coulomb, this function will add ParticleParameterOffset to the switching water and 'lambda_gc_coulomb'
        to the NonbondedForce.

        The CustomNonbondedForce should have the following energy expression:
            '(a/r6)^2-b/r6; r6=r^6;a=acoef(type1, type2);b=bcoef(type1, type2)'
            or
            'acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6;'

        :param system: openmm.System
            The system to be converted.
        :return: None
        """
        self.system = deepcopy(system)
        # check if the system is Charmm
        if self.system_type != "Charmm":
            raise ValueError("The system is not Charmm. Please check the system.")

        self.logger.info("Try to customise a native Charmm system")
        self.nonbonded_force = None
        self.custom_nonbonded_force = None
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if force.getName() == "NonbondedForce":
                self.nonbonded_force = force
            elif force.getName() == "CustomNonbondedForce":
                self.custom_nonbonded_force = force
        if self.custom_nonbonded_force.getNumGlobalParameters() != 0:
            raise ValueError("The CustomNonbondedForce should not have any global parameters. Please check the system.")

        energy_old = self.custom_nonbonded_force.getEnergyFunction()
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
        self.custom_nonbonded_force.setEnergyFunction(energy_expression)
        # Add per particle parameters
        self.custom_nonbonded_force.addPerParticleParameter("is_real")
        self.custom_nonbonded_force.addPerParticleParameter("is_switching")
        for atom_idx in range(self.custom_nonbonded_force.getNumParticles()):
            typ = self.custom_nonbonded_force.getParticleParameters(atom_idx)
            self.custom_nonbonded_force.setParticleParameters(atom_idx, [*typ, 1, 0])
        # Add global parameters
        self.custom_nonbonded_force.addGlobalParameter('softcore_alpha', 0.5)
        self.custom_nonbonded_force.addGlobalParameter('lambda_gc_vdw', 0.0) # lambda for vdw part of TI insertion/deletion
        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 0.0)  # lambda for coulomb part of TI insertion/deletion

        # set is_switching to 1.0 for the switching water
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force()
        for at_index in self.water_res_2_atom[self.switching_water]:
            parameters = list(self.custom_nonbonded_force.getParticleParameters(at_index))
            parameters[is_switching_index] = 1.0
            self.custom_nonbonded_force.setParticleParameters(at_index, parameters)

        # set ParticleParameters and add ParticleParameterOffset for the switching water
        for at_index in self.water_res_2_atom[self.switching_water]:
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.addParticleParameterOffset("lambda_gc_coulomb", at_index, charge, 0.0, 0.0)
            self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon) # remove charge

        # add Derivative
        # self.custom_nonbonded_force.addEnergyParameterDerivative('lambda_gc_vdw')

    def _customise_force_hybrid(self, system):
        """
        If the system is Hybrid, this function will add perParticleParameters **is_real** and **is_switching**
        to the self.custom_nonbonded_force (openmm.openmm.CustomNonbondedForce) for vdw.
        :param system: openmm.System
            The system to be converted.
        :return: None
        """
        self.system = deepcopy(system)
        # check if the system is Hybrid
        if self.system_type != "Hybrid":
            raise ValueError("The system is not Hybrid. Please check the system.")
        self.logger.info("Try to customise a native Hybrid system")
        self.custom_nonbonded_force = None
        self.nonbonded_force = None
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if force.getName() == "NonbondedForce":
                self.nonbonded_force = force
            elif force.getName() == "CustomNonbondedForce":
                self.custom_nonbonded_force = force

        energy_old = self.custom_nonbonded_force.getEnergyFunction()
        if energy_old != 'U_sterics;U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;lambda_alpha = new_interaction*(1-lambda_sterics_insert) + old_interaction*lambda_sterics_delete;lambda_sterics = core_interaction*lambda_sterics_core + new_interaction*lambda_sterics_insert + old_interaction*lambda_sterics_delete;core_interaction = delta(unique_old1+unique_old2+unique_new1+unique_new2);new_interaction = max(unique_new1, unique_new2);old_interaction = max(unique_old1, unique_old2);epsilonA = sqrt(epsilonA1*epsilonA2);epsilonB = sqrt(epsilonB1*epsilonB2);sigmaA = 0.5*(sigmaA1 + sigmaA2);sigmaB = 0.5*(sigmaB1 + sigmaB2);':
            raise ValueError(
                f"{energy_old} This energy expression in CustonNonbondedForce can not be converted. "
                f"Currently, grandfep only supports the system that is prepared by HybridTopologyFactory with Beutler softcore.")

        energy_expression = (
            "U_sterics;"
            "U_sterics = 4*epsilon*x*(x-1.0);"
            "x = (sigma/reff_sterics)^6;"
            
            # 6. Calculate damped distance (effective r)
            "reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);"
            
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
        self.custom_nonbonded_force.setEnergyFunction(energy_expression)
        self.custom_nonbonded_force.addPerParticleParameter("is_real")
        self.custom_nonbonded_force.addPerParticleParameter("is_switching")
        for atom_idx in range(self.custom_nonbonded_force.getNumParticles()):
            params = self.custom_nonbonded_force.getParticleParameters(atom_idx)
            self.custom_nonbonded_force.setParticleParameters(atom_idx, [*params, 1, 0]) # add is_real=1, is_switching=0
        # Add global parameters
        self.custom_nonbonded_force.addGlobalParameter('lambda_gc_vdw', 0.0)  # lambda for vdw part of TI insertion/deletion
        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 0.0)  # lambda for coulomb part of TI insertion/deletion

        # set is_switching to 1.0 for the switching water
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force()
        for at_index in self.water_res_2_atom[self.switching_water]:
            parameters = list(self.custom_nonbonded_force.getParticleParameters(at_index))
            parameters[is_switching_index] = 1.0
            self.custom_nonbonded_force.setParticleParameters(at_index, parameters)

        # set ParticleParameters and add ParticleParameterOffset for the switching water
        for at_index in self.water_res_2_atom[self.switching_water]:
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.addParticleParameterOffset("lambda_gc_coulomb", at_index, charge, 0.0, 0.0)
            self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon) # remove charge

        # add Derivative
        # self.custom_nonbonded_force.addEnergyParameterDerivative('lambda_gc_vdw')

    def _turn_off_vdw(self):
        """
        Turn off all vdw in self.custom_nonbonded_force.

        This function is only used for debugging purpose.

        :return: None
        """
        # All epsilon to 0.0
        for atom in self.topology.atoms():
            at_ind = atom.index
            sigmaA, epsilonA, sigmaB, epsilonB, unique_old, unique_new, is_real, is_switching = self.custom_nonbonded_force.getParticleParameters(at_ind)
            self.custom_nonbonded_force.setParticleParameters(at_ind, [sigmaA, 0.0*epsilonA, sigmaB, 0.0*epsilonB, unique_old, unique_new, is_real*0.0, is_switching])
        self.custom_nonbonded_force.updateParametersInContext(self.simulation.context)

    def set_ghost_list(self, ghost_list: list, check_system: bool = True) -> None:
        """
        Update the water residues in `ghost_list` to ghost.

        If `check_system` is True, `self.check_ghost_list` will be called to validate the consistency of self.ghost_list
        with `self.custom_nonbonded_force` and `self.nonbonded_force`. If the validation fails,
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

        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force()

        # set self.custom_nonbonded_force for vdw
        for res_index in self.ghost_list:
            if res_index not in self.water_res_2_atom:
                raise ValueError(f"The residue {res_index} is water.")
            for at_index in self.water_res_2_atom[res_index]:
                parameters = list(self.custom_nonbonded_force.getParticleParameters(at_index))
                parameters[is_real_index] = 1.0
                self.custom_nonbonded_force.setParticleParameters(at_index, parameters)

        for res_index in ghost_list:
            if res_index not in self.water_res_2_atom:
                raise ValueError(f"The residue {res_index} is not water.")
            for at_index in self.water_res_2_atom[res_index]:
                parameters = list(self.custom_nonbonded_force.getParticleParameters(at_index))
                parameters[is_real_index] = 0.0
                self.custom_nonbonded_force.setParticleParameters(at_index, parameters)

        # set self.nonbonded_force for coulomb
        for res_index in self.ghost_list:
            for at_index, wat_chg in zip(self.water_res_2_atom[res_index], self.wat_params['charge']):
                charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                self.nonbonded_force.setParticleParameters(at_index, wat_chg, sigma, epsilon)
        for res_index in ghost_list:
            for at_index in self.water_res_2_atom[res_index]:
                charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                self.nonbonded_force.setParticleParameters(at_index, charge * 0.0, sigma, epsilon)

        # update context
        self.nonbonded_force.updateParametersInContext(self.simulation.context)
        self.custom_nonbonded_force.updateParametersInContext(self.simulation.context)
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
            1. `is_real = 0.0` and `is_switching = 0.0` in `self.custom_nonbonded_force`
            2. Charge = 0.0 in `self.nonbonded_force`

        - Real and not switching water:
            1. `is_real = 1.0` in `self.custom_nonbonded_force`
            2. Charge = Proper_water_charge in `self.nonbonded_force`

        - Switching water:
            1. `is_real = 1.0` in `self.custom_nonbonded_force`
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

        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force()
        # check vdW in self.custom_nonbonded_force
        for res, at_index_list in self.water_res_2_atom.items():
            if res in self.ghost_list:
                # ghost water should have is_real = 0.0, is_switching = 0.0
                for at_index in at_index_list:
                    parameters = self.custom_nonbonded_force.getParticleParameters(at_index)
                    if parameters[is_real_index] > 1e-8:
                        raise ValueError(
                            f"The water {res} (real) atom {at_index} has is_real = {parameters[is_real_index]}.")
                    if parameters[is_switching_index] > 1e-8:
                        raise ValueError(
                            f"The water {res} (real) atom {at_index} has is_switching = {parameters[is_switching_index]}.")
            else:
                # real water should have is_real = 1.0
                for at_index in at_index_list:
                    parameters = self.custom_nonbonded_force.getParticleParameters(at_index)
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
        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force()
        # check vdW in self.custom_nonbonded_force
        for res, at_index_list in self.water_res_2_atom.items():
            if res != self.switching_water:
                # check is_switching should be 0.0
                for at_index in at_index_list:
                    parameters = self.custom_nonbonded_force.getParticleParameters(at_index)
                    if parameters[is_switching_index] > 1e-8:
                        self.logger.warning(
                            f"The water {res} at {at_index} has is_switching = {parameters[is_switching_index]}."
                        )
            else:
                # check is_switching should be 1.0, is_real should be 1.0
                for at_index in at_index_list:
                    parameters = self.custom_nonbonded_force.getParticleParameters(at_index)
                    if parameters[is_switching_index] < 0.99999999:
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

    def get_particle_parameter_index_cust_nb_force(self) -> tuple[int, int]:
        """
        Get the indices of `is_real` and `is_switching` parameters in the perParticleParameters
        of `self.custom_nonbonded_force`.

        Returns
        -------
        tuple of int
            A tuple `(is_real_index, is_switching_index)` indicating the indices of the corresponding
            parameters in `self.custom_nonbonded_force.getPerParticleParameterName(i)`.
        """
        # make sure we have the correct index for is_real and is_switching in perParticleParameters
        is_real_index = -1
        is_switching_index = -1
        for i in range(self.custom_nonbonded_force.getNumPerParticleParameters()):
            name = self.custom_nonbonded_force.getPerParticleParameterName(i)
            if name == "is_real":
                is_real_index = i
            elif name == "is_switching":
                is_switching_index = i
        if is_real_index == -1 or is_switching_index == -1:
            raise ValueError("The self.custom_nonbonded_force does not have is_real and is_switching parameters. Please check the system.")
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