import unittest
from pathlib import Path
import time

import numpy as np

from openmm import app, unit, openmm

from grandfep import utils, sampler

platform_ref = openmm.Platform.getPlatformByName('Reference')
nonbonded_Amber = {"nonbondedMethod": app.PME,
                   "nonbondedCutoff": 1.0 * unit.nanometer,
                   "constraints": app.HBonds,
                   }
nonbonded_Charmm = {"nonbondedMethod": app.PME,
                    "nonbondedCutoff": 1.2 * unit.nanometer,
                    "switchDistance" : 1.0 * unit.nanometer,
                    "constraints"    : app.HBonds
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
    error_msg = "".join([f"{at}\n    {f1}\n    {f2}\n" for at, f1, f2 in mis_match_list])
    return all_close_flag, mis_match_list, error_msg

def load_amber_sys(inpcrd_file, prmtop_file, nonbonded_settings):
    """
    Load Amber system from inpcrd and prmtop file.

    Parameters
    ----------
    inpcrd_file : str

    :param prmtop_file:

    :return: (inpcrd, prmtop, sys)
    Returns
    -------
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
        all_close_flag, mis_match_list, error_msg = match_force(force_4wat, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        # All Particles should be real, last water molecule should be switching
        self.assertEqual(baseGC.ghost_list, [])
        is_real_ind, is_switching_ind, custom_nb_force = baseGC.custom_nonbonded_force_list[0]
        self.assertEqual((2,3), (is_real_ind, is_switching_ind))
        for i in range(14):
            self.assertEqual(1.0, custom_nb_force.getParticleParameters(i)[is_real_ind])
            self.assertEqual(0.0, custom_nb_force.getParticleParameters(i)[is_switching_ind])
        for i in range(14,17):
            self.assertEqual(1.0, custom_nb_force.getParticleParameters(i)[is_real_ind])
            self.assertEqual(1.0, custom_nb_force.getParticleParameters(i)[is_switching_ind])


        print("## Set lambda_gc=0.0 for the switching water. Force should be the same as a 3-water system")
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
        all_close_flag, mis_match_list, error_msg = match_force(force_3wat, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Set one more water to ghost. Force should be the same as a 2-water system")
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
        all_close_flag, mis_match_list, error_msg = match_force(force_2wat, force[0:12])
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
        old_to_new_atom_map = {0: 0,
                               5: 8,    6: 9,   7: 10, # water res_index=1
                               8: 11,   9: 12, 10: 13, # water res_index=2
                               11: 14, 12: 15, 13: 16, # water res_index=3
                               14: 17, 15: 18, 16: 19, # water res_index=4
                               }
        old_to_new_core_atom_map = {0: 0,}
        h_factory = utils.HybridTopologyFactory(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,      # All atoms that should map from A to B
            old_to_new_core_atom_map, # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            softcore_LJ_v2=False)

        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform)


        print("## 1. Force should be the same before and after adding hybrid")
        inpcrd_4wat, prmtop_4wat, sys_4wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "06_solv.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "06_solv.prmtop", nonbonded_settings)
        energy_4wat, force_4wat = calc_energy_force(sys_4wat, prmtop_4wat.topology, inpcrd_4wat.positions, platform)

        # The forces should be the same
        self.assertEqual(len(force_4wat), 17)
        self.assertEqual(len(force_h), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force_4wat[1:], force_h[1:]) # There is dummy atom connected to atom 0 in state A
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")


        print("## 2. Force should be the same after customization")
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
        is_r, is_s,  custom_nb_force2= baseGC.custom_nonbonded_force_list[0]
        self.assertListEqual([is_r, is_s], [6,7])
        self.assertListEqual([], baseGC.get_ghost_list())
        baseGC.check_ghost_list()
        baseGC.check_switching()
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
        all_close_flag, mis_match_list, error_msg = match_force(force_h, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        all_close_flag, mis_match_list, error_msg = match_force(force_4wat[1:], force[1:]) # There is dummy atom connected to atom 0 in state A
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 3. Set lambda_gc=0.0 for the switching water. Force should be the same as a 3-water system")
        inpcrd_3wat, prmtop_3wat, sys_3wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "MOL_3wat.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "MOL_3wat.prmtop", nonbonded_settings)
        energy_3wat, force_3wat = calc_energy_force(sys_3wat, prmtop_3wat.topology, inpcrd_3wat.positions, platform)

        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        self.assertListEqual([], baseGC.get_ghost_list())
        baseGC.check_ghost_list()
        baseGC.check_switching()

        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        # The forces should be the same
        self.assertEqual(len(force_3wat), 14)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force_3wat[1:], force[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 4. Set one more water to ghost. Force should be the same as a 2-water system")
        inpcrd_2wat, prmtop_2wat, sys_2wat = load_amber_sys(
            base / "CH4_C2H6" /"lig0"/ "MOL_2wat.inpcrd",
            base / "CH4_C2H6" /"lig0"/ "MOL_2wat.prmtop", nonbonded_settings)
        energy_2wat, force_2wat = calc_energy_force(sys_2wat, prmtop_2wat.topology, inpcrd_2wat.positions, platform)

        # add a ghost water molecule
        baseGC.set_ghost_list([3])
        self.assertListEqual([3], baseGC.get_ghost_list())
        baseGC.check_ghost_list()
        baseGC.check_switching()
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        self.assertEqual(len(force_2wat), 11)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force_2wat[1:], force[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF_stateB(self):
        print()
        print("# Test HybridFF in stateB")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref
        base = Path(__file__).resolve().parent
        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig0" / "06_solv.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "CH4_C2H6" / "lig1" / "06_solv.inpcrd",
            base / "CH4_C2H6" / "lig1" / "06_solv.prmtop", nonbonded_settings)
        old_to_new_atom_map = {0: 0,
                               5: 8, 6: 9, 7: 10,  # water res_index=1
                               8: 11, 9: 12, 10: 13,  # water res_index=2
                               11: 14, 12: 15, 13: 16,  # water res_index=3
                               14: 17, 15: 18, 16: 19,  # water res_index=4
                               }
        old_to_new_core_atom_map = {0: 0}
        h_factory = utils.HybridTopologyFactory(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            softcore_LJ_v2=False)

        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
            platform=platform,
        )
        # set all lambdas to 1.0, and move to state B
        for lam in ["lambda_angles",
                    "lambda_bonds",
                    "lambda_torsions",
                    "lambda_electrostatics_core",
                    "lambda_electrostatics_delete",
                    "lambda_electrostatics_insert",
                    "lambda_sterics_core",
                    "lambda_sterics_delete",
                    "lambda_sterics_insert",]:
            baseGC.simulation.context.setParameter(lam, 1.0)

        print("## 1. Set lambda_gc=1.0. Force should be the same as a 4-water system.")
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        baseGC.simulation.context.setPositions(h_factory.hybrid_positions)
        baseGC.check_switching()
        baseGC.check_ghost_list()
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        energy_4wat, force_4wat = calc_energy_force(sys1, prmtop1.topology, inpcrd1.positions, platform)
        #             H   H   H   C   H   H   H
        index_hyb = [17, 18, 19, 20, 21, 22, 23,
                     5, 6, 7,
                     8, 9, 10,
                     11, 12, 13,
                     14, 15, 16]
        self.assertEqual(len(force_4wat), 20)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force[index_hyb], force_4wat[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 2. Set lambda_gc=1.0. Force should be the same as a 3-water system.")
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        inpcrd_3wat, prmtop_3wat, sys_3wat = load_amber_sys(
            base / "CH4_C2H6" /"lig1"/ "MOL_3wat.inpcrd",
            base / "CH4_C2H6" /"lig1"/ "MOL_3wat.prmtop", nonbonded_settings)
        energy_3wat, force_3wat = calc_energy_force(sys_3wat, prmtop_3wat.topology, inpcrd_3wat.positions, platform)
        index_hyb = [17, 18, 19, 20, 21, 22, 23,
                     5, 6, 7,
                     8, 9, 10,
                     11, 12, 13]
        self.assertEqual(len(force_3wat), 17)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force[index_hyb], force_3wat[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 3. Set one more water to ghost. Force should be the same as a 2-water system.")
        baseGC.set_ghost_list([3])
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        inpcrd_2wat, prmtop_2wat, sys_2wat = load_amber_sys(
            base / "CH4_C2H6" /"lig1"/ "MOL_2wat.inpcrd",
            base / "CH4_C2H6" /"lig1"/ "MOL_2wat.prmtop", nonbonded_settings)
        energy_2wat, force_2wat = calc_energy_force(sys_2wat, prmtop_2wat.topology, inpcrd_2wat.positions, platform)
        index_hyb = [17, 18, 19, 20, 21, 22, 23,
                     5, 6, 7,
                     8, 9, 10]
        self.assertEqual(len(force_2wat), 14)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force[index_hyb], force_2wat[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 4. Swap the coordinate of water 3 and 4. Force should not change.")
        pos[[11, 12, 13]], pos[[14, 15, 16]] = pos[[14, 15, 16]], pos[[11, 12, 13]]
        baseGC.simulation.context.setPositions(pos)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        all_close_flag, mis_match_list, error_msg = match_force(force[index_hyb], force_2wat[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 5. Set lambda_gc=1.0 for the switching water. Force should be the same as a 3-water system")
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        index_hyb = [17, 18, 19, 20, 21, 22, 23,
                     5, 6, 7,
                     8, 9, 10,
                     14, 15, 16]
        self.assertEqual(len(force_3wat), 17)
        self.assertEqual(len(force), 24)
        all_close_flag, mis_match_list, error_msg = match_force(force[index_hyb], force_3wat[1:])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_Charmm_psf(self):
        print()
        print("# Test Charmm system which is create from psf.createSystem")
        base = Path(__file__).resolve().parent

        nonbonded_settings = nonbonded_Charmm
        platform = platform_ref

        print("## Energy and force should be the same before and after customization")
        psf = app.CharmmPsfFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_12WAT.psf"))
        system = utils.load_sys(base / "KcsA_5VKE_SF/system_12WAT.xml.gz")
        topology = psf.topology
        pdb = app.PDBFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_12WAT.pdb"))
        topology.setPeriodicBoxVectors(
            np.array([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10]]) * unit.nanometer)
        energy_12wat, force_12wat = calc_energy_force(system, topology, pdb.positions, platform)
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            system,
            topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Charmm_psf.log",
            platform=platform,
        )  # Reference is necessary for exact force comparison

        self.assertEqual(baseGC.system_type, "Charmm")
        baseGC.simulation.context.setPositions(pdb.positions)
        self.assertDictEqual(baseGC.water_res_2_atom, {resi:[448+resi*3, 449+resi*3, 450+resi*3] for resi in range(42, 54)})

        # Can we get the same force before and after? baseGC has 4 water molecules
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        # The energy should be the same
        self.assertTrue(
            np.allclose(
                energy_12wat.value_in_unit(unit.kilojoule_per_mole),
                energy.value_in_unit(unit.kilojoule_per_mole)
            ), f"Energy mismatch: {energy_12wat} vs {energy}")
        # The forces should be the same for all atoms
        self.assertEqual(len(force_12wat), 610)
        self.assertEqual(len(force), 610)
        all_close_flag, mis_match_list, error_msg = match_force(force_12wat, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Remove 4 water, swithching water to 0 and 3 more water to ghost")
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        baseGC.set_ghost_list([50, 51, 52])
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        psf = app.CharmmPsfFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_8WAT.psf"))
        system = utils.load_sys(base / "KcsA_5VKE_SF/system_8WAT.xml.gz")
        topology = psf.topology
        pdb = app.PDBFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_8WAT.pdb"))
        topology.setPeriodicBoxVectors(
            np.array([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10]]) * unit.nanometer)
        energy_8wat, force_8wat = calc_energy_force(system, topology, pdb.positions, platform)
        # The forces should be the same for all atoms
        self.assertEqual(len(force_8wat), 598)
        self.assertEqual(len(force), 610)
        all_close_flag, mis_match_list, error_msg = match_force(force_8wat, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_Charmm_ff(self):
        print()
        print("# Test Charmm system which is create from ForceField.createSystem. ")
        base = Path(__file__).resolve().parent
        nonbonded_settings = nonbonded_Charmm
        platform = platform_ref

        ff = app.ForceField('charmm36.xml', 'charmm36/water.xml')
        psf = app.CharmmPsfFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_12WAT.psf"))
        pdb = app.PDBFile(str(base / "KcsA_5VKE_SF/step1_pdbreader_12WAT.pdb"))
        topology = psf.topology
        topology.setPeriodicBoxVectors(
            np.array([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10]]) * unit.nanometer)

        system = ff.createSystem(topology, **nonbonded_settings)

        print("## Expect ValueError when converting system which is created from ForceField.createSystem")
        with self.assertRaises(ValueError):
            baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
                system,
                topology,
                300 * unit.kelvin,
                1.0 / unit.picosecond,
                2.0 * unit.femtosecond,
                "test_base_Charmm_ff.log",
                platform=platform)

    def test_hybridFF_OPCwater(self):
        print()
        print("# Test HybridFF with OPC 4-site water")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent

        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "HSP90/water_leg/bcc_gaff2" / "2xab/solv_OPC/06_opc.inpcrd",
            base / "HSP90/water_leg/bcc_gaff2" / "2xab/solv_OPC/06_opc.prmtop", nonbonded_settings)
        inpcrd1, prmtop1, sys1 = load_amber_sys(
            base / "HSP90/water_leg/bcc_gaff2" / "2xjg/solv_OPC/06_opc.inpcrd",
            base / "HSP90/water_leg/bcc_gaff2" / "2xjg/solv_OPC/06_opc.prmtop", nonbonded_settings)
        old_to_new_atom_map = {}
        old_to_new_core_atom_map = {}
        pair_arr = np.loadtxt(base/"HSP90/water_leg/bcc_gaff2/pairs1.dat", dtype=int)
        for atA, atB in pair_arr:
            old_to_new_atom_map[atA-1] = atB-1
            old_to_new_core_atom_map[atA-1] = atB-1
        for i in range(41,233):
            old_to_new_atom_map[i] = i+3
        h_factory = utils.HybridTopologyFactory(
            sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
            old_to_new_atom_map,       # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            softcore_LJ_v2=False)

        energy_h, force_h = calc_energy_force(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            h_factory.hybrid_positions, platform)

        print("## 1. Force should be the same before and after adding hybrid")
        inpcrd, prmtop, sys = load_amber_sys(
            base / "HSP90/water_leg/bcc_gaff2" / "2xab/solv_OPC/06_opc.inpcrd",
            base / "HSP90/water_leg/bcc_gaff2" / "2xab/solv_OPC/06_opc.prmtop", nonbonded_settings)
        energy_allwat, force_allwat = calc_energy_force(sys, prmtop.topology, inpcrd.positions, platform)

        # The forces should be the same
        self.assertEqual(len(force_allwat), 233)
        self.assertEqual(len(force_h), 238)
        all_close_flag, mis_match_list, error_msg = match_force(
            force_allwat[41:233], force_h[41:233])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 2. Force should be the same before and after adding customization")
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            h_factory.hybrid_system,
            h_factory.omm_hybrid_topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
            platform=platform,
        )
        water_res_2_atom = {i:[i*4+37, i*4+38, i*4+39, i*4+40] for i in range(1, 49)}
        self.assertDictEqual(baseGC.water_res_2_atom, water_res_2_atom)

        # all lambdas are 0.0
        for lam in baseGC.simulation.context.getParameters():
            if lam.startswith("lambda"):
                self.assertAlmostEqual(baseGC.simulation.context.getParameter(lam), 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        baseGC.simulation.context.setPositions(h_factory.hybrid_positions)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        # The forces should be the same
        self.assertEqual(len(force), 238)
        all_close_flag, mis_match_list, error_msg = match_force(force_h, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_OPC_AmberFF(self):
        print()
        print("# Test AmberFF with OPC water")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref

        base = Path(__file__).resolve().parent
        inpcrd, prmtop, system = load_amber_sys(
            base / "Water_Chemical_Potential/OPC/water_opc.inpcrd",
            base / "Water_Chemical_Potential/OPC/water_opc.prmtop", nonbonded_settings)
        topology = prmtop.topology
        self.assertTrue(system.usesPeriodicBoundaryConditions())

        print("## Energy and force should be the same before and after customization")
        energy_ref, force_ref = calc_energy_force(system, topology, inpcrd.positions, platform)
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            system,
            topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Amber_OPC.log",
            platform=platform,
        )  # Reference platform is necessary for exact force comparison
        self.assertEqual(baseGC.system_type, "Amber")
        baseGC.simulation.context.setPositions(inpcrd.positions)

        # Can we get the same force before and after?
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        # The energy and force should be the same
        self.assertTrue(
            np.allclose(
                energy_ref.value_in_unit(unit.kilojoule_per_mole),
                energy.value_in_unit(unit.kilojoule_per_mole)
            ), f"Energy mismatch: {energy_ref} vs {energy}")
        self.assertEqual(len(force), 26644)
        self.assertEqual(len(force_ref), 26644)
        all_close_flag, mis_match_list, error_msg = match_force(force_ref, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## Set lambda_gc=0.0 for the switching water. Force should be the same as -1 water system")
        inpcrd, prmtop, system = load_amber_sys(
            base / "Water_Chemical_Potential/OPC/water_opc-1.inpcrd",
            base / "Water_Chemical_Potential/OPC/water_opc-1.prmtop", nonbonded_settings)
        topology = prmtop.topology
        energy_ref, force_ref = calc_energy_force(system, topology, inpcrd.positions, platform)

        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        self.assertTrue(
            np.allclose(
                energy_ref.value_in_unit(unit.kilojoule_per_mole),
                energy.value_in_unit(unit.kilojoule_per_mole)
            ), f"Energy mismatch: {energy_ref} vs {energy}")
        self.assertEqual(len(force), 26644)
        self.assertEqual(len(force_ref), 26640)
        all_close_flag, mis_match_list, error_msg = match_force(force_ref, force)
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF_protein(self):
        print()
        print("# Test Hybrid Protein System")
        nonbonded_settings = nonbonded_Amber
        platform = platform_ref
        base = Path(__file__).resolve().parent
        pdb = app.PDBFile(str(base / "CH4_C2H6/protein/system_hybrid.pdb"))
        system_h = utils.load_sys(base / "CH4_C2H6/protein/system_hybrid.xml.gz")
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            system_h,
            pdb.topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
            platform = platform
        )
        baseGC.simulation.context.setPositions(pdb.positions)
        self.assertTrue(209 in baseGC.water_res_2_atom)
        self.assertTrue(290 in baseGC.water_res_2_atom)
        self.assertTrue(291 not in baseGC.water_res_2_atom)

        print("## 1. Force should be the same before and after adding hybrid/customization")
        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6/protein/05-stateA.rst7",
            base / "CH4_C2H6/protein/05-stateA.prmtop", nonbonded_settings)
        energy_86, force_86 = calc_energy_force(sys0, prmtop0.topology, inpcrd0.positions)

        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 1.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 1.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)

        self.assertEqual(len(force_86), 3532)
        self.assertEqual(len(force), 3539)
        map = [i for i in range(3532) if i!=3281]
        all_close_flag, mis_match_list, error_msg = match_force(force_86[map], force[map])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 2. lambda_gc to 0.0, Force should be the same as a -1 water system")
        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6/protein/05-stateA-1.rst7",
            base / "CH4_C2H6/protein/05-stateA-1.prmtop", nonbonded_settings)
        energy_85, force_85 = calc_energy_force(sys0, prmtop0.topology, inpcrd0.positions)
        baseGC.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        baseGC.simulation.context.setParameter("lambda_gc_vdw", 0.0)
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        self.assertEqual(len(force_85), 3529)
        self.assertEqual(len(force), 3539)
        map = [i for i in range(3529) if i != 3281]
        all_close_flag, mis_match_list, error_msg = match_force(force_85[map], force[map])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

        print("## 2. Add one water to the ghost_list, Force should be the same as a -2 water system")
        inpcrd0, prmtop0, sys0 = load_amber_sys(
            base / "CH4_C2H6/protein/05-stateA-2.rst7",
            base / "CH4_C2H6/protein/05-stateA-2.prmtop", nonbonded_settings)
        energy_84, force_84 = calc_energy_force(sys0, prmtop0.topology, inpcrd0.positions)

        baseGC.set_ghost_list([289])
        state = baseGC.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos, energy, force = state.getPositions(asNumpy=True), state.getPotentialEnergy(), state.getForces(asNumpy=True)
        self.assertEqual(len(force_84), 3526)
        self.assertEqual(len(force), 3539)
        map = [i for i in range(3526) if i != 3281]
        all_close_flag, mis_match_list, error_msg = match_force(force_84[map], force[map])
        self.assertTrue(all_close_flag, f"In total {len(mis_match_list)} atom does not match. \n{error_msg}")

    def test_hybridFF_speed(self):
        print()
        print("# How long does it take to run updateParametersInContext()")
        nonbonded_settings = nonbonded_Amber
        base = Path(__file__).resolve().parent

        # Ligand in water system
        inpcrdA, prmtopA, sysA = load_amber_sys(
            base / "HSP90/water_leg/bcc_gaff2/2xab/06_solv.inpcrd",
            base / "HSP90/water_leg/bcc_gaff2/2xab/06_solv.prmtop", nonbonded_settings)
        inpcrdB, prmtopB, sysB = load_amber_sys(
            base / "HSP90/water_leg/bcc_gaff2/2xjg/06_solv.inpcrd",
            base / "HSP90/water_leg/bcc_gaff2/2xjg/06_solv.prmtop", nonbonded_settings)
        # Hybrid A and B
        mdp = utils.md_params_yml(base / "HSP90/water_leg/bcc_gaff2/mapping.yml")
        old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtopA.topology, prmtopB.topology,
                                                                               mdp.mapping_list)
        h_factory = utils.HybridTopologyFactory(
            sysA, inpcrdA.getPositions(), prmtopA.topology, sysB, inpcrdB.getPositions(), prmtopB.topology,
            old_to_new_atom_map,  # All atoms that should map from A to B
            old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            softcore_LJ_v2=False)
        system = h_factory.hybrid_system
        topology = h_factory.omm_hybrid_topology
        topology.setPeriodicBoxVectors(inpcrdA.boxVectors)
        system.setDefaultPeriodicBoxVectors(*inpcrdA.boxVectors)
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            system,
            topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
        )
        tick = time.time()
        baseGC.set_ghost_list([10, 11, 12], check_system=False)
        print(f"## Time for updating a lig-wat system {time.time() - tick:.3f} s")

        # Ligand protein water system
        system = utils.load_sys(base / "HSP90/protein_leg/system.xml.gz")
        topology = app.PDBFile(str(base / "HSP90/protein_leg/system.pdb")).topology
        baseGC_big = sampler.BaseGrandCanonicalMonteCarloSampler(
            system,
            topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Hybrid.log",
        )
        tick = time.time()
        baseGC_big.set_ghost_list([271, 272, 273], check_system=False)
        print(f"## Time for updating a lig-pro system {time.time() - tick:.3f} s")

        tick = time.time()
        is_r, is_j, custom_nb_force = baseGC_big.custom_nonbonded_force_list[0]
        custom_nb_force.updateParametersInContext(baseGC_big.simulation.context)
        print(f"## Time for updating C1 {time.time() - tick:.3f} s")

        tick = time.time()
        is_r, is_j, custom_nb_force = baseGC_big.custom_nonbonded_force_list[1]
        custom_nb_force.updateParametersInContext(baseGC_big.simulation.context)
        print(f"## Time for updating C2 {time.time() - tick:.3f} s")

        tick = time.time()
        is_r, is_j, custom_nb_force = baseGC_big.custom_nonbonded_force_list[2]
        custom_nb_force.updateParametersInContext(baseGC_big.simulation.context)
        print(f"## Time for updating C3 {time.time() - tick:.3f} s")

        baseGC_big.check_ghost_list()


if __name__ == '__main__':
    unittest.main()
