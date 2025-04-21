import unittest
from pathlib import Path

import numpy as np

from openmm import app, unit, openmm

from grandfep import utils, sampler


class MyTestCase(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent
        self.platform_ref = openmm.Platform.getPlatformByName('Reference')
        self.nonbonded_Amber = {"nonbondedMethod": app.PME,
                                "nonbondedCutoff": 1.0 * unit.nanometer,
                                "constraints": app.HBonds,
                                "hydrogenMass" : 3 * unit.amu,
                                }
        inpcrd = app.AmberInpcrdFile(str(base / "CH4_C2H6" / "lig0" / "06_solv.inpcrd"))
        prmtop = app.AmberPrmtopFile(str(base / "CH4_C2H6" / "lig0" / "06_solv.prmtop"),
                                     periodicBoxVectors=inpcrd.boxVectors)
        sys = prmtop.createSystem(**self.nonbonded_Amber)

        self.ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(
            system = sys,
            topology = prmtop.topology,
            temperature = 300 * unit.kelvin,
            collision_rate = 1 / (1 * unit.picoseconds),
            timestep = 1 * unit.femtoseconds,
            log = "test_ncgcmc.log",
            platform = self.platform_ref,
            water_resname = "HOH",
            water_O_name = "O",
            position = inpcrd.positions,
            chemical_potential = -6.09*unit.kilocalories_per_mole,
            standard_volume = 30.345*unit.angstroms**3,
            sphere_radius = 10.0*unit.angstroms,
            reference_atoms = [0],
        )
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
        state = self.ngcmc.simulation.context.getState(getPositions=True, getVelocities=True)

        o_index, h1_index, h2_index = self.ngcmc.water_res_2_atom[1]
        self.assertListEqual([o_index, h1_index, h2_index], [5,6,7])

        positions_old = state.getPositions(asNumpy=True)

        positions_new, vel = self.ngcmc.random_place_water(state, 1)
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
        positions_new, vel = self.ngcmc.random_place_water(state, 1, sphere_center = center)
        pos_o = positions_new[o_index].value_in_unit(unit.nanometer)
        r = np.linalg.norm(pos_o - center.value_in_unit(unit.nanometer))
        self.assertTrue(r <= self.ngcmc.sphere_radius.value_in_unit(unit.nanometer))
        self.check_water_bond_angle(positions_old, positions_new, o_index, h1_index, h2_index)

    def test_insertion_move_box(self):
        print()
        print("# insertion_move_box")
        self.ngcmc.set_ghost_list([1,2,3])
        self.ngcmc.simulation.context.setParameter("lambda_gc_coulomb", 0.0)
        self.ngcmc.simulation.context.setParameter("lambda_gc_vdw",     0.0)
        self.ngcmc.simulation.minimizeEnergy()
        self.ngcmc.simulation.step(100)
        self.ngcmc.insertion_move_box(
            np.linspace(0,1,11),
            np.linspace(0,1,11), 10)



if __name__ == '__main__':
    unittest.main()
