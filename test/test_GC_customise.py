import unittest
from pathlib import Path

from openmm import app, unit, openmm

from grandfep import utils, sampler



class MyTestCase(unittest.TestCase):
    def test_AmberFF(self):
        base = Path(__file__).resolve().parent
        inpcrd = app.AmberInpcrdFile(base / "CH4_C2H6" /"lig0"/ "06_solv.inpcrd")
        prmtop = app.AmberPrmtopFile(base / "CH4_C2H6" /"lig0"/ "06_solv.prmtop",
                                     periodicBoxVectors = inpcrd.boxVectors)
        topology = prmtop.topology
        system = prmtop.createSystem(nonbondedMethod=app.PME,
                                     nonbondedCutoff=1 * unit.nanometer,
                                     constraints=app.HBonds,
                                     )
        self.assertTrue(system.usesPeriodicBoundaryConditions())
        baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
            system,
            topology,
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
            "test_base_Amber.log"
        )
        self.assertEqual(baseGC.system_type, "Amber")
        baseGC.sim.context.setPositions(inpcrd.positions)

        # Can we get the same force before and after?
        baseGC.set_water_switch(4, True)
        print()

if __name__ == '__main__':
    unittest.main()
