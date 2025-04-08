import unittest
from pathlib import Path

import numpy as np
from docutils.parsers.rst.directives import positive_int

from openmm import app, unit, openmm

from grandfep import utils, sampler


def calc_energy_force(system, topology, positions, platform=openmm.Platform.getPlatform('Reference')):
    """
    Calculate energy and force for the system
    :param system: openmm.System
    :param topology: openmm.Topology
    :param positions: openmm.Vec3
    :return: energy, force
    """
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    state = simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
    energy = state.getPotentialEnergy()
    force = state.getForces(asNumpy=True)
    return energy, force

def match_force(force1, force2):
    """
    Check if the force is the same
    :param force1: state.getForces(asNumpy=True)
    :param force2: state.getForces(asNumpy=True)
    :return: bool
    """
    all_close_flag = True
    mis_match_list = []
    for at_index in range(len(force1)):
        at_flag = np.allclose(force1[at_index], force2[at_index])
        if not at_flag:
            all_close_flag = False
            mis_match_list.append([at_index, force1[at_index], force2[at_index]])
    return all_close_flag, mis_match_list

class MyTestCase(unittest.TestCase):
    def test_AmberFF(self):
        nonbonded_settings = {"nonbondedMethod": app.PME,
                              "nonbondedCutoff": 1.0 * unit.nanometer,
                              "constraints": app.HBonds,
                              }
        platform = openmm.Platform.getPlatformByName('Reference')

        base = Path(__file__).resolve().parent
        inpcrd = app.AmberInpcrdFile(str(base / "CH4_C2H6" /"lig0"/ "06_solv.inpcrd"))
        prmtop = app.AmberPrmtopFile(str(base / "CH4_C2H6" /"lig0"/ "06_solv.prmtop"),
                                     periodicBoxVectors = inpcrd.boxVectors)
        topology = prmtop.topology
        system = prmtop.createSystem(**nonbonded_settings)
        self.assertTrue(system.usesPeriodicBoundaryConditions())

        print("# Energy and force should be the same before and after customization")
        energy_4wat, force_4wat = calc_energy_force(system, topology, inpcrd.positions, platform) # this is the reference value
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            system,
            topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Amber.log",
            platform=platform,
        ) # Reference is necessary for exact force comparison
        self.assertEqual(baseGC.system_type, "Amber")
        baseGC.sim.context.setPositions(inpcrd.positions)

        # Can we get the same force before and after? baseGC has 4 water molecules
        self.assertDictEqual(baseGC.water_res_2_atom, {1:[5,6,7], 2:[8,9,10], 3:[11,12,13], 4:[14,15,16]})
        state = baseGC.sim.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        # The energy should be the same
        self.assertTrue(
            np.allclose(
                energy_4wat.value_in_unit(unit.kilojoule_per_mole),
                energy.value_in_unit(unit.kilojoule_per_mole)
            ), f"Energy mismatch: {energy_4wat} vs {energy}")
        # The forces should be the same for 0-16 atoms
        all_close_flag, mis_match_list = match_force(force_4wat, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. {mis_match_list}")

        # All Particles should satisfy : is_real = 1.0, is_switching = 0.0
        self.assertEqual(baseGC.ghost_list, [])
        is_real_ind, is_switching_ind = baseGC.get_particle_parameter_index_cust_nb_force()
        self.assertEqual((2,3), (is_real_ind, is_switching_ind))
        for i in range(17):
            self.assertEqual(1.0, baseGC.custom_nonbonded_force.getParticleParameters(i)[is_real_ind])
            self.assertEqual(0.0, baseGC.custom_nonbonded_force.getParticleParameters(i)[is_switching_ind])

        print("# Force should be the same if we set a water molecule as switching while keeping lambda=1")
        baseGC.set_water_switch(4, True)
        baseGC.check_switching()
        baseGC.sim.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.sim.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.sim.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        # The energy should be the same
        self.assertTrue(
            np.allclose(
                energy_4wat.value_in_unit(unit.kilojoule_per_mole),
                energy.value_in_unit(unit.kilojoule_per_mole)
            ), f"Energy mismatch: {energy_4wat} vs {energy}")
        # The forces should be the same for 0-16 atoms
        all_close_flag, mis_match_list = match_force(force_4wat, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. {mis_match_list}")

        print("# Set a water to ghost. Force should be the same as -1 water system")
        inpcrd = app.AmberInpcrdFile(base / "CH4_C2H6" /"lig0"/ "MOL_3wat.inpcrd")
        prmtop = app.AmberPrmtopFile(base / "CH4_C2H6" /"lig0"/ "MOL_3wat.prmtop",
                                     periodicBoxVectors = inpcrd.boxVectors)
        top_3wat = prmtop.topology
        sys_3wat = prmtop.createSystem(**nonbonded_settings)
        pos_3wat = inpcrd.positions
        energy_3wat, force_3wat = calc_energy_force(sys_3wat, top_3wat, pos_3wat, platform)

        baseGC.set_water_switch(4, False)
        baseGC.set_ghost_list([4], True)
        self.assertEqual(baseGC.switching_water, -1)
        baseGC.check_ghost_list()
        state = baseGC.sim.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        all_close_flag, mis_match_list = match_force(force_3wat, force[0:14])
        # print force pair line by line if not match
        error_msg = "".join([f"{at}\n    {f1}\n    {f2}\n"  for at, f1, f2 in mis_match_list])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("# Set a water to switching and lambda=0.0, force should be the same as -1 water system")
        baseGC.set_water_switch(4, True)
        self.assertEqual(baseGC.ghost_list, [])

        self.assertEqual(baseGC.ghost_list, [])
        is_real_ind, is_switching_ind = baseGC.get_particle_parameter_index_cust_nb_force()
        for i in range(14):
            self.assertEqual(1.0, baseGC.custom_nonbonded_force.getParticleParameters(i)[is_real_ind])
            self.assertEqual(0.0, baseGC.custom_nonbonded_force.getParticleParameters(i)[is_switching_ind])
        for i in range(14, 17):
            self.assertEqual(1.0, baseGC.custom_nonbonded_force.getParticleParameters(i)[is_real_ind])
            self.assertEqual(1.0, baseGC.custom_nonbonded_force.getParticleParameters(i)[is_switching_ind])
        self.assertEqual(3, baseGC.nonbonded_force.getNumParticleParameterOffsets())


        ###### For debugging purpose
        # set charge to 0.0 for water with res_index 4
        baseGC._remove_water_charge(4)
        for i in range(3):
            global_name, at_index, chargeS, sigmaS, epsilonS = baseGC.nonbonded_force.getParticleParameterOffset(i)
            print(global_name, at_index, chargeS, sigmaS, epsilonS)
        ######

        baseGC.sim.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.sim.context.setParameter("lambda_gc_vdw", 0.0)
        self.assertAlmostEqual(baseGC.sim.context.getParameter("lambda_gc_coulomb"), 0.0)
        self.assertAlmostEqual(baseGC.sim.context.getParameter("lambda_gc_vdw"), 0.0)
        state = baseGC.sim.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        all_close_flag, mis_match_list = match_force(force_3wat, force[0:14])
        # print force pair line by line if not match
        error_msg = "".join([f"{at}\n    {f1}\n    {f2}\n"  for at, f1, f2 in mis_match_list])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")


if __name__ == '__main__':
    unittest.main()
