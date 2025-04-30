import unittest
import copy
from pathlib import Path

import numpy as np

from openmm import app, unit, openmm
from pytraj import get_velocity

from grandfep import utils, sampler

l_vdw = [0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.67, 0.89, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
l_chg = [0.0, 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]



class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parent
        platform_ref = openmm.Platform.getPlatformByName('Reference')
        self.nonbonded_Amber = {"nonbondedMethod": app.PME,
                                "nonbondedCutoff": 1.0 * unit.nanometer,
                                "constraints": app.HBonds,
                                }
        inpcrd = app.AmberInpcrdFile(str(self.base / "CH4_C2H6/lig0/05_solv.rst7"))
        prmtop = app.AmberPrmtopFile(str(self.base / "CH4_C2H6/lig0/05_solv.prmtop"),
                                     periodicBoxVectors=inpcrd.boxVectors)
        sys = prmtop.createSystem(**self.nonbonded_Amber)

        self.args_dict = {
            "system": sys,
            "topology": prmtop.topology,
            "temperature": 300 * unit.kelvin,
            "collision_rate": 1 / (1 * unit.picoseconds),
            "timestep": 2 * unit.femtoseconds,
            "log": "test_ncgcmc_IO.log",
            "platform": platform_ref,
            "water_resname": "HOH",
            "water_O_name": "O",
            "position": inpcrd.positions,
            "chemical_potential": -6.09 * unit.kilocalories_per_mole,
            "standard_volume": 30.345 * unit.angstroms ** 3,
            "sphere_radius": 7.0 * unit.angstroms,
            "reference_atoms": [0],
            "rst_file": self.base / "CH4_C2H6/lig0/test_IO.rst7",
            "dcd_file": self.base / "CH4_C2H6/lig0/test_IO.dcd",
            "append_dcd": False,
            "jsonl_file": self.base / "CH4_C2H6/lig0/test_IO.jsonl",
        }
    def test_init(self):
        ngcmc = sampler.NoneqGrandCanonicalMonteCarloSampler(**self.args_dict)
        ngcmc.report_dcd()
        ngcmc.set_ghost_from_jsonl(self.base / "CH4_C2H6/lig0/05_solv_init.jsonl")
        ngcmc.report_dcd()

        line_list = ['{"GC_count": 0, "ghost_list": []}\n',
                     '{"GC_count": 50, "ghost_list": [11, 12, 13]}\n'
                     ]
        with open(self.base / "CH4_C2H6/lig0/test_IO.jsonl") as f:
            lines = f.readlines()
        self.assertListEqual(line_list, lines)


if __name__ == '__main__':
    unittest.main()
