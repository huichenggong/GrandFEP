import unittest
import copy
from pathlib import Path

import numpy as np

from openmm import app, unit, openmm
from pytraj import get_velocity

from grandfep import utils, sampler


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parent
        self.platform_ref = openmm.Platform.getPlatformByName('Reference')
        self.nonbonded_Amber = {"nonbondedMethod": app.PME,
                                "nonbondedCutoff": 1.0 * unit.nanometer,
                                "constraints": app.HBonds,
                                "hydrogenMass" : 3 * unit.amu,
                                }
        inpcrd = app.AmberInpcrdFile(str(self.base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd"))
        prmtop = app.AmberPrmtopFile(str(self.base / "CH4_C2H6" / "lig0" / "06_solv.prmtop"),
                                     periodicBoxVectors=inpcrd.boxVectors)
        sys = prmtop.createSystem(**self.nonbonded_Amber)

        self.args_dict = {
            "system" : sys,
            "topology" : prmtop.topology,
            "temperature" : 300 * unit.kelvin,
            "collision_rate" : 1 / (1 * unit.picoseconds),
            "timestep" : 1 * unit.femtoseconds,
            "log" : "test_ncgcmc.log",
            "platform" : self.platform_ref,
            "water_resname" : "HOH",
            "water_O_name" : "O",
            "position" : inpcrd.positions,
            "chemical_potential" : -6.09 * unit.kilocalories_per_mole,
            "standard_volume" : 30.345 * unit.angstroms ** 3,
            "sphere_radius" : 10.0 * unit.angstroms,
            "reference_atoms" : [0],
        }
        self.args_dict2 = copy.copy(self.args_dict)
        self.args_dict2["position"] = app.AmberInpcrdFile(str(self.base / "CH4_C2H6" / "lig0" / "06_solv_shift.inpcrd")).positions

        self.args_dict3 = copy.copy(self.args_dict)
        inpcrd = app.AmberInpcrdFile(
            str(self.base / "Water_Chemical_Potential/TIP3P/water.inpcrd"))
        prmtop = app.AmberPrmtopFile(
            str(self.base / "Water_Chemical_Potential/TIP3P/water.prmtop"),
            periodicBoxVectors=inpcrd.boxVectors)
        sys = prmtop.createSystem(**self.nonbonded_Amber)
        self.args_dict3["system"] = sys
        self.args_dict3["topology"] = prmtop.topology
        self.args_dict3["position"] = inpcrd.positions
        self.args_dict3["platform"] = openmm.Platform.getPlatformByName('CUDA')

    def check_water_bond_angle(self, positions_old, positions_new, o_index, h1_index, h2_index):
        """
        O-H bond length should be kept the same
        H-O-H angle should be kept the same
        """
        for at_index in [h1_index, h2_index]:
            bond_vec_old = positions_old[at_index] - positions_old[o_index]
            bond_vec_new = positions_new[at_index] - positions_new[o_index]
            bond_length_old = np.linalg.norm(bond_vec_old)
            bond_length_new = np.linalg.norm(bond_vec_new)
            self.assertAlmostEqual(bond_length_old, bond_length_new)
            # print("bond length", bond_length_old, bond_length_new)

        angle_list = []
        for pos in [positions_old, positions_new]:
            vec1 = pos[h1_index] - pos[o_index]
            vec2 = pos[h2_index] - pos[o_index]
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            angle_list.append(angle)
        self.assertAlmostEqual(angle_list[0], angle_list[1])
        # print("angle", angle_list[0], angle_list[1])

    def test_random_place_water(self):
        print()
        print("# insert_random_water_box, position should be shifted, bond_length and angle should be kept the same")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict)
        state = ngcmc.simulation.context.getState(getPositions=True, getVelocities=True)

        o_index, h1_index, h2_index = ngcmc.water_res_2_atom[1]
        self.assertListEqual([o_index, h1_index, h2_index], [5,6,7])

        positions_old = state.getPositions(asNumpy=True)

        positions_new, vel = ngcmc.random_place_water(state, 1)
        # O sites inside the box
        pos_o = positions_new[o_index]
        self.assertFalse(np.any(pos_o == positions_old[o_index])) # xyz should all be different
        box_v = state.getPeriodicBoxVectors(asNumpy=True)
        for i in range(3):
            self.assertTrue(0*unit.nanometer < pos_o[i])
            self.assertTrue(pos_o[i] < box_v[i][i])

        self.check_water_bond_angle(positions_old, positions_new, o_index, h1_index, h2_index)

        # O sites inside the sphere
        center = np.array([1.5, 1.5, 1.5]) * unit.nanometers
        positions_new, vel = ngcmc.random_place_water(state, 1, sphere_center = center)
        pos_o = positions_new[o_index].value_in_unit(unit.nanometer)
        r = np.linalg.norm(pos_o - center.value_in_unit(unit.nanometer))
        self.assertTrue(r <= ngcmc.sphere_radius.value_in_unit(unit.nanometer))
        self.check_water_bond_angle(positions_old, positions_new, o_index, h1_index, h2_index)

    def test_move_box(self):
        print()
        print("# insertion_move_box")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict)
        ngcmc.set_ghost_list([1,2,3])
        ngcmc.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        ngcmc.simulation.context.setParameter("lambda_gc_vdw",     0.0)
        ngcmc.simulation.minimizeEnergy()
        ngcmc.simulation.step(100)
        l_vdw = [0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.67, 0.89, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        l_chg = [0.0, 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print(ngcmc.move_insertion_box(l_vdw,
                                            l_chg,
                                            10))

        ngcmc.set_ghost_list([1, 2])
        ngcmc.simulation.minimizeEnergy()
        ngcmc.simulation.step(100)
        print(ngcmc.move_deletion_box(l_vdw[::-1],
                                      l_chg[::-1],
                                      10))

    def test_get_sphere_center(self):
        print()
        print("# Find center")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict2)
        pos = ngcmc.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        print(pos[0])
        center = ngcmc.get_sphere_center(pos)
        self.assertTrue(np.all(np.isclose(pos[0], center)))

    def test_get_water_state(self):
        print()
        print("# FIX PBC")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict2)
        pos = ngcmc.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        water_state_dict, dist_all_o = ngcmc.get_water_state(pos)
        self.assertTrue(
            np.all(np.isclose(
                dist_all_o, np.array([0.4513818 , 0.43114845, 0.39188307, 0.33323369])
            )))
        self.assertDictEqual(water_state_dict, {1: 1, 2: 1, 3: 1, 4: 1})

    def test_insert_empty(self):
        print()
        print("# Insert a water into an empty box")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict3)
        ghost_list = [res_index for res_index in ngcmc.water_res_2_O][:-1]
        ngcmc.set_ghost_list(ghost_list)
        ngcmc.simulation.context.setVelocitiesToTemperature(200 * unit.kelvin)
        state = ngcmc.simulation.context.getState(getPositions=True, getVelocities=True)
        pos_old = state.getPositions(asNumpy=True)
        vel_old = state.getVelocities(asNumpy=True)

        l_vdw = [0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.67, 0.89, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        l_chg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        accept, acc_prob, protocol_work, protocol_work_list, n_water = ngcmc.move_insertion_box(
            l_vdw, l_chg, 10)
        state = ngcmc.simulation.context.getState(getPositions=True, getVelocities=True)
        pos_new = state.getPositions(asNumpy=True)
        vel_new = state.getVelocities(asNumpy=True)
        if accept:
            self.assertEqual(n_water, 1)
            close_pos = np.sum(np.isclose(pos_old, pos_new))
            self.assertTrue(close_pos<50, f"Number of pos_xyz that are very close : {close_pos}")
            close_vel = np.sum(np.isclose(vel_old, vel_new))
            self.assertTrue(close_vel<10, f"Number of vel_xyz that are very close : {close_vel}")
        else:
            self.assertEqual(n_water, 0)
            self.assertTrue(np.all(np.isclose(pos_old, pos_new)), "Move is rejected, but the position has been changed")
            self.assertTrue(np.all(np.isclose(vel_old, vel_new)), "Move is rejected, but the velocity has been changed")
        print(accept, acc_prob, protocol_work, n_water)

    def test_delete_empty(self):
        print()
        print("# Deletion but with small Adam")
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict3)
        ngcmc.Adam_box = -35

        ngcmc.simulation.context.setVelocitiesToTemperature(200 * unit.kelvin)
        state = ngcmc.simulation.context.getState(getPositions=True, getVelocities=True)
        pos_old = state.getPositions(asNumpy=True)
        vel_old = state.getVelocities(asNumpy=True)

        l_vdw = [0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.67, 0.89, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        l_chg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        accept, acc_prob, protocol_work, protocol_work_list, n_water = ngcmc.move_deletion_box(
            l_vdw[::-1], l_chg[::-1], 10)
        state = ngcmc.simulation.context.getState(getPositions=True, getVelocities=True)
        pos_new = state.getPositions(asNumpy=True)
        vel_new = state.getVelocities(asNumpy=True)
        if accept:
            self.assertEqual(n_water, 6659)
            close_pos = np.sum(np.isclose(pos_old, pos_new))
            self.assertTrue(close_pos < 50, f"Number of pos_xyz that are very close : {close_pos}")
            close_vel = np.sum(np.isclose(vel_old, vel_new))
            self.assertTrue(close_vel < 10, f"Number of vel_xyz that are very close : {close_vel}")
        else:
            self.assertEqual(n_water, 6660)
            self.assertTrue(np.all(np.isclose(pos_old, pos_new)), "Move is rejected, but the position has been changed")
            self.assertTrue(np.all(np.isclose(vel_old, vel_new)), "Move is rejected, but the velocity has been changed")
        print(accept, acc_prob, protocol_work, n_water)


if __name__ == '__main__':
    unittest.main()
