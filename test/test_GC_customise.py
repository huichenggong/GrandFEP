import unittest
from pathlib import Path

import numpy as np
from docutils.parsers.rst.directives import positive_int

from openmm import app, unit, openmm

from grandfep import utils, sampler

platform_ref = openmm.Platform.getPlatformByName('Reference')
nonbonded_Amber = {"nonbondedMethod": app.PME,
                   "nonbondedCutoff": 1.0 * unit.nanometer,
                   "constraints": app.HBonds,
                   }

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
    for at_index, (f1, f2) in enumerate(zip(force1, force2)):
        at_flag = np.allclose(f1, f2)
        if not at_flag:
            all_close_flag = False
            mis_match_list.append([at_index, f1, f2])
    return all_close_flag, mis_match_list

def load_amber_sys(inpcrd_file, prmtop_file, nonbonded_settings):
    """
    Load Amber system from inpcrd and prmtop file.
    :param inpcrd_file:

    :param prmtop_file:

    :return: (inpcrd, prmtop, sys)
        inpcrd: openmm.app.AmberInpcrdFile
        prmtop: openmm.app.AmberPrmtopFile
        sys: openmm.System
    """
    inpcrd = app.AmberInpcrdFile(str(inpcrd_file))
    prmtop = app.AmberPrmtopFile(str(prmtop_file),
                                 periodicBoxVectors=inpcrd.boxVectors)
    sys = prmtop.createSystem(**nonbonded_settings)
    return inpcrd, prmtop, sys

class MyTestCase(unittest.TestCase):
    def test_AmberFF(self):
        print()
        print("# Test AmberFF, Can we get the correct force when changing real/ghost and switching water")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent
        inpcrd, prmtop, system = load_amber_sys(
            base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig0" / "06_solv.prmtop", nonbonded_settings)
        topology = prmtop.topology
        self.assertTrue(system.usesPeriodicBoundaryConditions())

        print("## Energy and force should be the same before and after customization")
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
        baseGC.simulation.context.setPositions(inpcrd.positions)
        self.assertDictEqual(baseGC.water_res_2_atom, {1:[5,6,7], 2:[8,9,10], 3:[11,12,13], 4:[14,15,16]})

        # Can we get the same force before and after? baseGC has 4 water molecules
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        # The energy should be the same
        self.assertTrue(
            np.allclose(
                energy_4wat.value_in_unit(unit.kilojoule_per_mole),
                energy.value_in_unit(unit.kilojoule_per_mole)
            ), f"Energy mismatch: {energy_4wat} vs {energy}")
        # The forces should be the same for 0-16 atoms
        self.assertEqual(len(force_4wat), 17)
        self.assertEqual(len(force), 17)
        all_close_flag, mis_match_list = match_force(force_4wat, force)
        error_msg = "".join([f"{at}\n    {f1}\n    {f2}\n"  for at, f1, f2 in mis_match_list])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        # All Particles should be real, last water molecule should be switching
        self.assertEqual(baseGC.ghost_list, [])
        is_real_ind, is_switching_ind = baseGC.get_particle_parameter_index_cust_nb_force()
        self.assertEqual((2,3), (is_real_ind, is_switching_ind))
        for i in range(14):
            self.assertEqual(1.0, baseGC.custom_nonbonded_force.getParticleParameters(i)[is_real_ind])
            self.assertEqual(0.0, baseGC.custom_nonbonded_force.getParticleParameters(i)[is_switching_ind])
        for i in range(14,17):
            self.assertEqual(1.0, baseGC.custom_nonbonded_force.getParticleParameters(i)[is_real_ind])
            self.assertEqual(1.0, baseGC.custom_nonbonded_force.getParticleParameters(i)[is_switching_ind])


        print("## Set lambda=0.0 for the switching water. Force should be the same as -1 water system")
        inpcrd_3wat, prmtop_3wat, sys_3wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "MOL_3wat.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "MOL_3wat.prmtop", nonbonded_settings)
        energy_3wat, force_3wat = calc_energy_force(sys_3wat, prmtop_3wat.topology, inpcrd_3wat.positions, platform)

        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        baseGC.check_switching()
        # The forces should be the same for 0-13 atoms
        self.assertEqual(len(force_3wat), 14)
        self.assertEqual(len(force), 17)
        all_close_flag, mis_match_list = match_force(force_3wat, force)
        error_msg = "".join([f"{at}\n    {f1}\n    {f2}\n"  for at, f1, f2 in mis_match_list])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Set one more water to ghost. Force should be the same as -2 water system")
        inpcrd_2wat, prmtop_2wat, sys_2wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "MOL_2wat.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "MOL_2wat.prmtop", nonbonded_settings)
        energy_2wat, force_2wat = calc_energy_force(sys_2wat, prmtop_2wat.topology, inpcrd_2wat.positions, platform)


        baseGC.set_ghost_list([3]) # water residues: 1:real, 2:real, 3:ghost, 4:switching
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        baseGC.check_switching()
        # The forces should be the same for 0-11 atoms
        self.assertEqual(len(force_2wat), 11)
        self.assertEqual(len(force), 17)
        all_close_flag, mis_match_list = match_force(force_2wat, force[0:12])
        error_msg = "".join([f"{at}\n    {f1}\n    {f2}\n"  for at, f1, f2 in mis_match_list])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF(self):
        print()
        print("# Test HybridFF, Can we get the correct force when changing real/ghost and switching water")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig0" / "06_solv.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "CH4_C2H6" / "lig1" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig1" / "06_solv.prmtop", nonbonded_settings)
        old_to_new_atom_map = {0: 0, 5: 8, 6: 9, 7: 10, 8: 11,
                               9: 12, 10: 13, 11: 14, 12: 15, 13: 16,
                               14: 17, 15: 18, 16: 19}
        old_to_new_core_atom_map = {0: 0}
        h_factory = utils.HybridTopologyFactory(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map, old_to_new_core_atom_map,
            use_dispersion_correction=True,
            softcore_LJ_v2=False)

        h_factory.hybrid_system,
        h_factory.omm_hybrid_topology,

        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform)

        print("## Force should be the same before and after adding hybrid")
        inpcrd_4wat, prmtop_4wat, sys_4wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "06_solv.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "06_solv.prmtop", nonbonded_settings)
        energy_4wat, force_4wat = calc_energy_force(sys_4wat, prmtop_4wat.topology, inpcrd_4wat.positions, platform)

        # The forces should be the same
        self.assertEqual(len(force_4wat), 17)
        self.assertEqual(len(force_h), 24)
        all_close_flag, mis_match_list = match_force(force_4wat[1:], force_h[1:]) # There is dummy atom connected to atom 0 in state A
        error_msg = "".join([f"{at}\n    {f1}\n    {f2}\n"  for at, f1, f2 in mis_match_list])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Force should be the same after customization")
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
            platform=platform,
        )
        self.assertEqual(baseGC.system_type, "Hybrid")
        baseGC.simulation.context.setPositions(h_factory.hybrid_positions)
        self.assertDictEqual(baseGC.water_res_2_atom, {1:[5,6,7], 2:[8,9,10], 3:[11,12,13], 4:[14,15,16]})
        # all lambdas are 0.0
        for lam in baseGC.simulation.context.getParameters():
            if lam.startswith("lambda"):
                self.assertAlmostEqual(baseGC.simulation.context.getParameter(lam), 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        # The forces should be the same
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list = match_force(force_h, force)
        error_msg = "".join([f"{at}\n    {f1}\n    {f2}\n"  for at, f1, f2 in mis_match_list])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Set lambda=0.0 for the switching water. Force should be the same as -1 water system")
        inpcrd_3wat, prmtop_3wat, sys_3wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "MOL_3wat.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "MOL_3wat.prmtop", nonbonded_settings)

        #### Debugging
        for f in sys_3wat.getForces():
            if f.getName() == "NonbondedForce":
                nonbonded = f
                break
        # all epsilon to 0.0
        for atom in prmtop_3wat.topology.atoms():
            at_ind = atom.index
            charge, sigma, epsilon = list(nonbonded.getParticleParameters(at_ind))
            print(charge, sigma, epsilon)
            nonbonded.setParticleParameters(at_ind, charge, sigma, 0.0*epsilon)


        energy_3wat, force_3wat = calc_energy_force(sys_3wat, prmtop_3wat.topology, inpcrd_3wat.positions, platform)


        #### Debugging
        baseGC._turn_off_vdw()
        ####


        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        # baseGC.check_switching()




        # The forces should be the same for 0:14 atoms
        self.assertEqual(len(force_3wat), 14)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list = match_force(force_3wat, force)
        error_msg = "".join([f"{at}\n    {f1}\n    {f2}\n"  for at, f1, f2 in mis_match_list])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")









if __name__ == '__main__':
    unittest.main()
