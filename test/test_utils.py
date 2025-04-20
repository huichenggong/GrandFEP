import unittest
from pathlib import Path

import numpy as np

from openmm import app, unit, openmm

from grandfep import utils




class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.base_path = Path(__file__).resolve().parent

    def test_find_reference_atom_indices(self):
        top = self.base_path / "KcsA_5VKE_SF/step1_pdbreader_12WAT.psf"
        topology, _ = utils.load_top(top)
        ref_atoms_list = [{"res_id":"71", "res_name":"GLU", "atom_name":"O"}]
        ref_atoms = utils.find_reference_atom_indices(topology, ref_atoms_list)
        self.assertListEqual(ref_atoms, [21, 164, 307, 450])



if __name__ == '__main__':
    unittest.main()
