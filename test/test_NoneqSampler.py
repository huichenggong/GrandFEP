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
            timestep = 4 * unit.femtoseconds,
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

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
