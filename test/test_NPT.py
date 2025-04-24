import unittest
from pathlib import Path
from sys import stdout

import numpy as np

from openmm import app, unit, openmm

from grandfep import utils, sampler

platform_ref = openmm.Platform.getPlatformByName('Reference')

nonbonded_Amber = {"nonbondedMethod": app.PME,
                   "nonbondedCutoff": 1.0 * unit.nanometer,
                   "constraints": app.HBonds,
                   }

class MyTestCase(unittest.TestCase):
    def test_initNPT(self):
        print()
        print("# init an NPT sampler with pure OPC water")
        base = Path(__file__).resolve().parent
        inpcrd = app.AmberInpcrdFile(str(base / "Water_Chemical_Potential/OPC/water_opc.inpcrd"))
        prmtop = app.AmberPrmtopFile(str(base / "Water_Chemical_Potential/OPC/water_opc.prmtop"),
                                     periodicBoxVectors=inpcrd.boxVectors)
        system = prmtop.createSystem(**nonbonded_Amber)
        system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))

        npt = sampler.NPTSampler(
            system=system,
            topology=prmtop.topology,
            temperature=300 * unit.kelvin,
            collision_rate=1 / (1 * unit.picoseconds),
            timestep=1 * unit.femtoseconds,
            log="test_npt.log",
            rst_file="opc_npt_output.rst7",
            dcd_file="opc_npt_output.dcd"
        ) # use default CUDA platform
        npt.check_temperature()

        state_reporter = app.StateDataReporter(
            "opc_npt_output.csv", 1000, step=True, time=True, temperature=True, density=True)
        npt.simulation.reporters.append(state_reporter)

        state_reporter_std = app.StateDataReporter(
            stdout, 5000, step=True, temperature=True, density=True, speed=True)
        npt.simulation.reporters.append(state_reporter_std)

        npt.simulation.context.setPositions(inpcrd.positions)
        npt.simulation.minimizeEnergy()
        npt.simulation.context.setVelocitiesToTemperature(npt.temperature/2)
        npt.simulation.step(25000)

        npt.report_rst()



if __name__ == '__main__':
    unittest.main()
