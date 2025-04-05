from copy import deepcopy
import logging
import copy

import numpy as np

from openmm import unit, app, openmm
from openmmtools.integrators import BAOABIntegrator

from .. import utils

class BaseGrandCanonicalMonteCarloSampler:
    """
    Base class for Grand Canonical Monte Carlo (GCMC) sampling.
    This class provides the basic object to initiate the forces so that water can be added (real to Dum) or removed (Dum to real).
    """
    def __init__(self, system, topology, temperature,
                 collision_rate, timestep,
                 log,
                 water_resname="HOH", water_O_name="O"):
        """

        :param system: openmm.openmm.System
            Currently, the system generated from the following methods (Or with the equivalent forces) are supported:
        :param topology: openmm.app.topology.Topology
            The name of water must be HOH. Currently, 3-point and 4-point water models are supported.
        :param temperature:
            Reference temperature for the system, with proper unit.
        """

        # prepare logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                                    "%m-%d %H:%M:%S"))
        self.logger.addHandler(file_handler)
        self.logger.info("Initialize BaseGrandCanonicalMonteCarloSampler")

        self.system = None
        self.topology = topology
        self.simulation = None
        self.nonbonded_force = None        # coulomb
        self.custom_nonbonded_force = None # vdw
        self.kBT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.temperature = temperature
        self.Adam_box = None
        self.Adam_GCMC = None


        self.ghost_list = []  # list of water residue index that are ghost

        # preparation based on the topology
        self.logger.info("Check topology")
        self.num_of_points_water = self._check_water_points(water_resname)
        self.water_res_2_atom = None
        self.water_res_2_O = None
        self._find_all_water(water_resname, water_O_name)

        # preparation based on the system
        self.logger.info("Prepare system")
        self.system_type = self._check_system(system)
        self.custom_nonbonded_force = None
        self.nonbonded_force = None
        self.wat_params = None
        self._get_water_parameters(water_resname, system)

        if self.system_type == "Amber":
            self._customise_force_amber(system)
        elif self.system_type == "Charmm":
            self._customise_force_charmm(system)
        elif self.system_type == "Hybrid":
            self._customise_force_hybrid(system)
        else:
            raise ValueError(f"The system ({self.system_type}) cannot be customized. Please check the system.")

        # prepare integrator, simulation
        self.logger.info("Prepare integrator and simulation")
        self.compound_integrator = openmm.CompoundIntegrator()
        integrator = BAOABIntegrator(temperature, collision_rate, timestep)
        self.compound_integrator.addIntegrator(integrator) # for EQ run
        integrator = BAOABIntegrator(temperature, collision_rate, timestep)
        self.compound_integrator.addIntegrator(integrator) # for NCMC run

        self.sim = app.Simulation(self.topology, self.system, self.compound_integrator)

    def _find_all_water(self, resname, water_O_name):
        """
        Check topology setup self.water_res_2_atom and self.water_res_2_O.
        :return: None
        """
        # check if the system has water
        self.water_res_2_atom = {}
        self.water_res_2_O = {}
        for res in self.topology.residues():
            if res.name == resname:
                self.water_res_2_atom[res.index] = []
                self.water_res_2_O[res.index] = []
                for atom in res.atoms():
                    self.water_res_2_atom[res.index].append(atom.index)
                    if atom.name == water_O_name:
                        self.water_res_2_O[res.index].append(atom.index)

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
        :param system: openmm.System
            The system to be checked.
        :return: str
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
        elif nb_force_flag and c_nb_force_flag and not c_bond_flag and not c_angle_flag and not c_torsion_flag:
            system_type =  "Charmm"
        else:
            raise ValueError("The system is not supported. Please check the force in the system.")

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
        Get
        :param resname:
        :return:
        """
        for f in system.getForces():
            if f.getName() == "NonbondedForce":
                nonbonded_force = f
                break

        wat_params = []  # Store parameters in a list
        for residue in self.topology.residues():
            if residue.name == resname:
                for atom in residue.atoms():
                    # Store the parameters of each atom
                    atom_params = nonbonded_force.getParticleParameters(atom.index)
                    wat_params.append({'charge' : atom_params[0],
                                       'sigma' : atom_params[1],
                                       'epsilon' : atom_params[2]})
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
            self.custom_nonbonded_force.addParticle([sigma, epsilon, 1.0, 1.0]) # add
            self.nonbonded_force.setParticleParameters(at_index, charge, sigma, 0.0 * epsilon) # remove

        # Exceptions will not be changed in NonbondedForce, but will the corresponding pairs need to be excluded in CustomNonbondedForce
        for exception_idx in range(self.nonbonded_force.getNumExceptions()):
            [i, j, charge_product, sigma, epsilon] = self.nonbonded_force.getExceptionParameters(exception_idx)
            # Add vdw exclusion to CustomNonbondedForce
            self.custom_nonbonded_force.addExclusion(i, j)

        self.system.addForce(self.custom_nonbonded_force)

    def _customise_force_charmm(self, system):
        """
        In Charmm, NonbondedForce handles electrostatics, and CustomNonbondedForce handles vdW. This function will add
        lambda parameters to the CustomNonbondedForce to handle the switching off of interactions for certain water.
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
                "U = lambda_vdw * ( (a/reff6)^2-b/reff6 );"
                "reff6 = sigma6 * ((softcore_alpha * (1-lambda_vdw) + (r/sigma)^6));" # reff^6
                "sigma6 = a^2 / b;"  # sigma^6
                "lambda_vdw = is_real1 * is_real2 * lambda_switch;"
                "lambda_switch = (1 - switch_indicator) + lambda_gc_vdw * switch_indicator;"
                "switch_indicator = max(is_switching1, is_switching2);"
                "a = acoef(type1, type2);"  # a = 2 * epsilon^0.5 * sigma^6
                "b = bcoef(type1, type2);"  # b = 4 * epsilon * sigma^6
            )
        elif energy_old == 'acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6;':
            energy_expression = (
                "U = lambda_vdw * ( (a/reff6)^2-b/reff6 );"
                "reff6 = sigma6 * ((softcore_alpha * (1-lambda_vdw) + (r/sigma)^6));" # reff^6
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
            self.custom_nonbonded_force.setParticleParameters(atom_idx, [*typ, 1, 1])
        # Add global parameters
        self.custom_nonbonded_force.addGlobalParameter('softcore_alpha', 0.5)
        self.custom_nonbonded_force.addGlobalParameter('lambda_gc_vdw', 0.0) # lambda for vdw part of TI insertion/deletion

    def _customise_force_hybrid(self, system):
        """
        If the system is Hybrid, this function will add perParticleParameters lambda_gc to the CustomNonbondedForce for
        vdw.
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
            "U_sterics = lambda_vdw * 4*epsilon*x*(x-1.0);"
            "x = (sigma/reff_sterics)^6;"
            "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;"
            "reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);"
            "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"
            "lambda_alpha = new_interaction*(1-lambda_sterics_insert) + old_interaction*lambda_sterics_delete + (1 - lambda_vdw);"
            "lambda_vdw = is_real1 * is_real2 * lambda_switch;"
            "lambda_sterics = core_interaction*lambda_sterics_core + new_interaction*lambda_sterics_insert + old_interaction*lambda_sterics_delete;"
            "core_interaction = delta(unique_old1+unique_old2+unique_new1+unique_new2);"
            "new_interaction = max(unique_new1, unique_new2);"
            "old_interaction = max(unique_old1, unique_old2);"
            "lambda_switch = (1 - switch_indicator) + lambda_gc_vdw * switch_indicator;"
            "switch_indicator = max(is_switching1, is_switching2);" # whether the two particles are switching in GC
            "epsilonA = sqrt(epsilonA1*epsilonA2);"
            "epsilonB = sqrt(epsilonB1*epsilonB2);"
            "sigmaA = 0.5*(sigmaA1 + sigmaA2);"
            "sigmaB = 0.5*(sigmaB1 + sigmaB2);"
        )
        # Add per particle parameters
        self.custom_nonbonded_force.addPerParticleParameter("is_real")
        self.custom_nonbonded_force.addPerParticleParameter("is_switching")
        for atom_idx in range(self.custom_nonbonded_force.getNumParticles()):
            typ = self.custom_nonbonded_force.getParticleParameters(atom_idx)
            self.custom_nonbonded_force.setParticleParameters(atom_idx, [*typ, 1, 1])
        # Add global parameters
        self.custom_nonbonded_force.addGlobalParameter('softcore_alpha', 0.5)
        self.custom_nonbonded_force.addGlobalParameter('lambda_gc_vdw', 0.0)  # lambda for vdw part of TI insertion/deletion

    def set_ghost_list(self, ghost_list, from_system=False):
        """
        Set all the water to real and set all the water in given ghost_list to ghost.
        :param ghost_list:
            Water residue index that should be set to ghost.
        :param from_system: bool
            If True, during the resetting of old ghost water to real, the system will be checked to verify the ghost list.
        :return:
        """
        pass

    def get_ghost_list(self, from_system=False):
        """
        Get the ghost list. If from_system is True, system will be checked to verify the ghost list.
        :param from_system: bool
            If True, the ghost list will be generated from the system.
        :return: list
        """
        pass

    def set_water_lambda(self, res_index, lambda_coulomb, lambda_vdw):
        """
        Set the lambda value for the given water residue index.
        :param res_index: int
            The residue index of the water.
        :param lambda_coulomb: float
            The lambda value for the Coulomb interaction.
        :param lambda_vdw: float
            The lambda value for the Lennard-Jones interaction.
        """
        pass



