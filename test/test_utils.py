import unittest
from pathlib import Path

import numpy as np

from openmm import app, unit, openmm

from grandfep import utils




class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.base_path = Path(__file__).resolve().parent

    def test_find_reference_atom_indices(self):
        print()
        print("# Find atom indices from residue and atom_name")
        top = self.base_path / "KcsA_5VKE_SF/step1_pdbreader_12WAT.psf"
        topology, _ = utils.load_top(top)
        ref_atoms_list = [{"res_id":"71", "res_name":"GLU", "atom_name":"O"}]
        ref_atoms = utils.find_reference_atom_indices(topology, ref_atoms_list)
        self.assertListEqual(ref_atoms, [21, 164, 307, 450])

    def test_random_rotation_matrix(self):
        print()
        print("# Generate uniformly distributed random rotation matrix")
        res_new = []
        res_old = []
        vec_init = np.array([0, 0, 1])
        for i in range(1_000):

            rot_matrix = utils.random_rotation_matrix()
            res_new.append(np.dot(rot_matrix, vec_init))

            rot_matrix = utils.random_rotation_matrix_protoms()
            res_old.append(np.dot(rot_matrix, vec_init))

        np.save(self.base_path / "output/rotation_matrix_new.npy", res_new)
        np.save(self.base_path / "output/rotation_matrix_old.npy", res_old)

    def test_free_e_analysis(self):
        print()
        print("# Use pymbar to estimate the free energy")
        file_list = [self.base_path / f"Water_Chemical_Potential/OPC/MBAR/{i}/md.log" for i in range(20)]
        keyword="Reduced Energy U_i(x):"
        separator=","

        analysis = utils.FreeEAnalysis(file_list, keyword, separator, 10)
        self.assertAlmostEqual(analysis.temperature.value_in_unit(unit.kelvin), 300.0)

        print()
        analysis.print_uncorrelate()
        res_all = {"MBAR": analysis.mbar_U_all(),
                   "BAR": analysis.bar_U_all()}

        analysis.print_res_all(res_all)

    def test_md_params_yml(self):
        print()
        print("# Load a yaml file which has the MD parameters")
        mdp = utils.md_params_yml(self.base_path/"Water_Chemical_Potential/OPC/multidir/0/md.yml")
        print(mdp.get_system_setting())
        print(mdp)

    def test_reporters(self):
        print()
        print("# Test dcdReporter")

        print()
        print("# Test rst7Reporter")


if __name__ == '__main__':
    unittest.main()
