from copy import deepcopy
import logging

import numpy as np

from openmm import unit, app, openmm
from openmmtools.integrators import BAOABIntegrator
from pandas.core.indexers import is_list_like_indexer

from .. import utils

class BaseGrandCanonicalMonteCarloSampler:
    """
    Base class for Grand Canonical Monte Carlo (GCMC) sampling.
    This class provides the basic object to initiate the forces so that water can be added (real to Dum) or removed (Dum to real).
    """
    def __init__(self, system, topology, temperature,
                 collision_rate, timestep,
                 log,
                 platform=openmm.Platform.getPlatform('CUDA'),
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

        self.sim = app.Simulation(self.topology, self.system, self.compound_integrator, platform)

        self.switching_water = None

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
        Get the charge of water and save it in self.wat_params['charge'].
        :param resname: str
        :return: None
        """
        nonbonded_force = None
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
                    wat_params.append({'charge' : atom_params[0]})
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
        self.custom_nonbonded_force.addGlobalParameter('lambda_gc_vdw', 1.0) # lambda for vdw part of TI insertion/deletion
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
        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 1.0) # lambda for coulomb part of TI insertion/deletion

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
        self.custom_nonbonded_force.addGlobalParameter('lambda_gc_vdw', 1.0) # lambda for vdw part of TI insertion/deletion
        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 1.0)  # lambda for coulomb part of TI insertion/deletion

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
        self.custom_nonbonded_force.addGlobalParameter('lambda_gc_vdw', 1.0)  # lambda for vdw part of TI insertion/deletion
        self.nonbonded_force.addGlobalParameter('lambda_gc_coulomb', 1.0)  # lambda for coulomb part of TI insertion/deletion


    def set_ghost_list(self, ghost_list, check_system=False):
        """
        Set all the water to real and set all the water in given ghost_list to ghost.
        :param ghost_list:
            Residue index of water that should be set to ghost.
        :param check_system: bool
            If True, check all the water particles in self.custom_nonbonded_force and self.nonbonded_force to make sure
            the ghost_list is correctly set.
        :return:
        """
        if self.switching_water in ghost_list:
            self.set_water_switch(self.switching_water, False, check_system=check_system)

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
            for at_index, wat_param in zip(self.water_res_2_atom[res_index], self.wat_params):
                wat_chg = wat_param['charge']
                charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                self.nonbonded_force.setParticleParameters(at_index, wat_chg, sigma, epsilon)
        for res_index in ghost_list:
            for at_index in self.water_res_2_atom[res_index]:
                charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                self.nonbonded_force.setParticleParameters(at_index, 0.0, sigma, epsilon)

        # update context
        self.nonbonded_force.updateParametersInContext(self.sim.context)
        self.custom_nonbonded_force.updateParametersInContext(self.sim.context)
        self.ghost_list = ghost_list

        if check_system:
            self.check_ghost_list()

    def get_ghost_list(self, check_system=False):
        """
        Get the ghost list. If from_system is True, system will be checked to verify the ghost list.
        :param from_system: bool
            If True, the ghost list will be generated from the system.
        :return: list
            A copy of the ghost list.
        """
        if check_system:
            self.check_ghost_list()
        return deepcopy(self.ghost_list)

    def check_ghost_list(self):
        """
        Loop over water particles in self.custom_nonbonded_force and self.nonbonded_force to make sure the ghost_list is correct.
        All ghost water should have is_real = 0.0, is_switching = 0.0 in custom_nonbonded_force,
        0.0 charge in nonbonded_force, no ParticleParameterOffsets or chargeScale=0.0 in ParticleParameterOffsets.
        All real water should have is_real = 1.0, custom_nonbonded_force, and proper charge in nonbonded_force
        Raise ValueError if any of the above is not satisfied.
        :return: None
        """
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
        particle_parameter_offset_dict = self.get_particle_parameter_offset_dict()
        for res, at_index_list in self.water_res_2_atom.items():
            if res in self.ghost_list:
                # ghost water should have charge = 0.0, and (no ParticleParameterOffsets, or chargeScale=0.0)
                for at_index in at_index_list:
                    charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                    if abs(charge.value_in_unit(unit.elementary_charge)) > 1e-8:
                        raise ValueError(
                            f"The water {res} at {at_index} has charge = {charge}."
                        )
                    if at_index in particle_parameter_offset_dict:
                        param_offset_index, param_name, _at_index, charge_scale, sigma_scale, epsilon_scale = particle_parameter_offset_dict[at_index]
                        if param_name != "lambda_gc_coulomb":
                            raise ValueError(f"Water {res} (ghost) at {at_index} has a nonbonded force parameter offset {param_name}. ")
                        if charge_scale > 1e-8:
                            raise ValueError(f"Water {res} (ghost) at {at_index} has non-zero ParticleParameterOffset {particle_parameter_offset_dict[at_index]} .")
            else:
                # real water should have proper charge
                for at_index, wat_param in zip(self.water_res_2_atom[res], self.wat_params):
                    wat_chg = wat_param['charge']
                    charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                    if not np.allclose(charge.value_in_unit(unit.elementary_charge), wat_chg.value_in_unit(unit.elementary_charge)):
                        raise ValueError(
                            f"The water {res} at {at_index} has charge = {charge}, should be {wat_chg.value_in_unit(unit.elementary_charge)}."
                        )
                    if at_index in particle_parameter_offset_dict:
                        param_offset_index, param_name, _at_index, charge_scale, sigma_scale, epsilon_scale = particle_parameter_offset_dict[at_index]
                        if param_name != "lambda_gc_coulomb":
                            raise ValueError(f"Water {res} (real) at {at_index} has a nonbonded force parameter offset {param_name}. ")

    def check_switching(self):
        """
        Loop over all particles in self.custom_nonbonded_force and self.nonbonded_force to make sure the one water given
        by self.switching_water is the only one that is switching.
        :return:
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
                # either no ParticleParameterOffsets or chargeScale=0.0
                for at_index in at_index_list:
                    if at_index in particle_parameter_offset_dict:
                        param_offset_index, param_name, _at_index, charge_scale, sigma_scale, epsilon_scale = particle_parameter_offset_dict[at_index]
                        if param_name != "lambda_gc_coulomb":
                            raise ValueError(f"Water {res} (no switching) at {at_index} has a nonbonded force parameter offset {param_name}. ")
                        if charge_scale > 1e-8:
                            raise ValueError(f"Water {res} (no switching) at {at_index} has non-zero ParticleParameterOffset {particle_parameter_offset_dict[at_index]} .")
            else:
                # check ParticleParameterOffsets for the water to chargeScale=-chg, sigmaScale=0.0, epsilonScale=0.0
                for at_index, wat_param in zip(self.water_res_2_atom[res], self.wat_params):
                    wat_chg = wat_param['charge']
                    if at_index in particle_parameter_offset_dict:
                        param_offset_index, param_name, _at_index, charge_scale, sigma_scale, epsilon_scale = particle_parameter_offset_dict[at_index]
                        if param_name != "lambda_gc_coulomb":
                            raise ValueError(f"Water {res} (is switching) at {at_index} has a nonbonded force parameter offset {param_name}. ")
                        if charge_scale > abs(-wat_chg.value_in_unit(unit.elementary_charge)):
                            raise ValueError(f"Water {res} (is switching) at {at_index} has wrong ParticleParameterOffset for chargeScale {particle_parameter_offset_dict[at_index]} .")
                        if sigma_scale > 1e-8 or epsilon_scale > 1e-8:
                            raise ValueError(f"Water {res} (is switching) at {at_index} has non-zero ParticleParameterOffset for vdw {particle_parameter_offset_dict[at_index]} .")
                    else:
                        raise ValueError(f"Water {res} (is switching) at {at_index} does not have a ParticleParameterOffset.")

    def set_water_switch(self, res_index, on, check_system=True):
        """
        Set PerParticleParameter for the specified water. The PerParticleParameter is_switching and is_real in
        custom_nonbonded_force will be changed so that vdw interaction on this water can be controlled by the
        lambda_gc_vdw. The ParticleParameterOffsets in nonbonded_force will be updated so that coulomb interaction on
        this water can be controlled by lambda_gc_coulomb.
        :param res_index: int
            The residue index of the water.
        :param on: bool
            If True, the certain water will be set to is_switching = 1.0, is_real=0, which means the water will be controlled by
            lambda_gc_vdw and lambda_gc_coulomb.
            If False, the certain water will be set to is_switching = 0.0.
        :param check_system: bool, default True
            If True, the system will be checked to make sure there is only one is_switching water, when turning on the
            water, and there is no is_switching water, when turning off the water. If the check fails, system will be
            corrected, and a warning will be raised.
        """

        is_real_index, is_switching_index = self.get_particle_parameter_index_cust_nb_force()

        # update custom_nonbonded_force for vdw
        if on:
            if res_index in self.ghost_list:
                self.ghost_list.remove(res_index) # switching water should not be in ghost water
            for at_index in self.water_res_2_atom[res_index]:
                parameters = list(self.custom_nonbonded_force.getParticleParameters(at_index))
                parameters[is_real_index] = 1.0
                parameters[is_switching_index] = 1.0
                self.custom_nonbonded_force.setParticleParameters(at_index, parameters)
        else:
            for at_index in self.water_res_2_atom[res_index]:
                parameters = list(self.custom_nonbonded_force.getParticleParameters(at_index))
                parameters[is_switching_index] = 0.0
                self.custom_nonbonded_force.setParticleParameters(at_index, parameters)

        # update nonbonded_force for coulomb
        if on: # this water should have proper charge
            for at_index, wat_param in zip(self.water_res_2_atom[res_index], self.wat_params):
                wat_chg = wat_param['charge']
                charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
                self.nonbonded_force.setParticleParameters(at_index, wat_chg, sigma, epsilon)

        # Set the ParticleParameterOffsets for the water in nonbonded_force
        particle_parameter_offset_dict = self.get_particle_parameter_offset_dict()
        if on:
            # set the ParticleParameterOffsets for the water to chargeScale=-chg, sigmaScale=0.0, epsilonScale=0.0
            for at_index, wat_param in zip(self.water_res_2_atom[res_index], self.wat_params):
                wat_chg = wat_param['charge']
                if at_index in particle_parameter_offset_dict:
                    param_offset_index, param_name, _at_index, charge_scale, sigma_scale, epsilon_scale = particle_parameter_offset_dict[at_index]
                    if param_name != "lambda_gc_coulomb":
                        raise ValueError(f"Water {res_index} at {at_index} has a nonbonded force parameter offset {param_name}. ")
                    self.nonbonded_force.setParticleParameterOffset(param_offset_index, param_name, _at_index, -wat_chg, 0.0, 0.0)
                else:
                    self.nonbonded_force.addParticleParameterOffset("lambda_gc_coulomb", at_index, -wat_chg, 0.0, 0.0)
        else:
            # set the ParticleParameterOffsets for the water to chargeScale=0.0, sigmaScale=0.0, epsilonScale=0.0
            for at_index in self.water_res_2_atom[res_index]:
                if at_index in particle_parameter_offset_dict:
                    param_offset_index, param_name, _at_index, charge_scale, sigma_scale, epsilon_scale = particle_parameter_offset_dict[at_index]
                    if param_name != "lambda_gc_coulomb":
                        raise ValueError(f"Water {res_index} at {at_index} has a nonbonded force parameter offset {param_name}. ")
                    self.nonbonded_force.setParticleParameterOffset(param_offset_index, param_name, _at_index, 0.0, 0.0, 0.0)


        self.nonbonded_force.updateParametersInContext(self.sim.context)
        self.custom_nonbonded_force.updateParametersInContext(self.sim.context)
        if on:
            self.switching_water = res_index
        else:
            self.switching_water = -1
        if check_system:
            if on:
                self.check_switching()
            else:
                self.check_switching()

    def _remove_water_charge(self, res_index):
        """
        Remove the charge of the water in the system. The charge will be set to 0.0 in self.nonbonded_force. This
        function is only for debugging purpose.
        :param res_index: int
            The residue index of the water.
        :return: None
        """
        for at_index in self.water_res_2_atom[res_index]:
            charge, sigma, epsilon = self.nonbonded_force.getParticleParameters(at_index)
            self.nonbonded_force.setParticleParameters(at_index, 0.0 * charge, sigma, epsilon)
        self.nonbonded_force.updateParametersInContext(self.sim.context)

    def get_particle_parameter_index_cust_nb_force(self):
        """
        Get the index of is_real and is_switching in perParticleParameters of custom_nonbonded_force.
        :return: is_real_index, is_switching_index
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

    def get_particle_parameter_offset_dict(self):
        """
        Get the ParticleParameterOffsets in nonbonded_force.
        :return: dict
            A dictionary of ParticleParameterOffsets in nonbonded_force. The key is the atom index, and the value is
            [param_offset_index, global_parameter_name, at_index, chargeScale, sigmaScale, epsilonScale].
        """
        particle_parameter_offset_dict = {}  # {at_index: [param_offset_index, global_parameter_name, at_index, chargeScale, sigmaScale, epsilonScale]}
        for param_offset_index in range(self.nonbonded_force.getNumParticleParameterOffsets()):
            offset = self.nonbonded_force.getParticleParameterOffset(param_offset_index)
            at_index = offset[1]
            particle_parameter_offset_dict[at_index] = [param_offset_index, *offset]
        return particle_parameter_offset_dict