
import logging
import sys
import unittest
from pathlib import Path
from typing import Union, Tuple

import numpy as np
from openmm import app, unit, openmm

from grandfep import utils, sampler
# from grandfep.sampler.base import TerminalFlipMC

nonbonded_Amber = {"nonbondedMethod": app.PME,
                   "nonbondedCutoff": 1.0 * unit.nanometer,
                   "constraints": app.HBonds,
                   }

platform_ref = openmm.Platform.getPlatformByName('Reference')

def load_amber_sys(inpcrd_file: Union[str, Path],
                   prmtop_file: Union[str, Path],
                   nonbonded_settings: dict) -> Tuple[app.AmberInpcrdFile, app.AmberPrmtopFile, openmm.System]:
    """
    Load Amber system from inpcrd and prmtop file.
    """
    inpcrd = app.AmberInpcrdFile(str(inpcrd_file))
    prmtop = app.AmberPrmtopFile(str(prmtop_file),
                                 periodicBoxVectors=inpcrd.boxVectors)
    sys = prmtop.createSystem(**nonbonded_settings)
    return inpcrd, prmtop, sys


class TestRotateTerminal(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parent
        self.output = self.base / "output"
        self.output.mkdir(exist_ok=True)

        inpcrd, prmtop, system = load_amber_sys(
            self.base / "thro/lig_prep/1a_pose_2/07_tip3p.inpcrd",
            self.base / "thro/lig_prep/1a_pose_2/07_tip3p.prmtop",
            nonbonded_Amber,
        )
        self.topology = prmtop.topology

        integrator = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin,
                                                     1.0 * unit.picosecond**-1,
                                                     2.0 * unit.femtosecond
                                                     )
        simulation = app.Simulation(self.topology, system, integrator, platform_ref)
        simulation.context.setPositions(inpcrd.positions)

        # terminal_list: atoms 19,20 define the C15-C16 rotation axis;
        # atoms 21-30 are the phenyl ring that rotates.
        terminal_list = [
            [180, [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]
        ]

        logger = logging.getLogger("test_TerminalFlipMC")
        logger.setLevel(logging.INFO)
        # log to a file
        logger.addHandler(logging.FileHandler(self.output / "test_TerminalFlipMC.log"))
        kBT = 298.15 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R
        self.flipmc = sampler.TerminalFlipMC(
            simulation=simulation,
            topology=self.topology,
            kBT=kBT,
            logger=logger,
            terminal_list=terminal_list,
        )

    def _get_positions(self):
        state = self.flipmc.simulation.context.getState(getPositions=True)
        return state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    def _save_pdb(self, filename):
        state = self.flipmc.simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        positions = state.getPositions()
        path = self.output / filename
        with open(path, "w") as f:
            app.PDBFile.writeFile(self.topology, positions, f)
        print(f"  Saved: {path}")

    def test_rotate_terminal(self):
        print()
        print("# Test rotate_terminal: 180° rotation around C15-C16 bond")
        angle, terminal = self.flipmc.terminal_list[0]
        pivot_idx = terminal[1]   # C16 – rotation centre
        axis_idx  = terminal[0]   # C15 – axis start
        mobile    = terminal[2:]  # phenyl ring atoms

        pos_before = self._get_positions()
        self._save_pdb("rotate_terminal_before.pdb")

        self.flipmc.rotate_terminal(0)

        pos_after = self._get_positions()
        self._save_pdb("rotate_terminal_after.pdb")

        # Axis and pivot atoms must NOT move
        np.testing.assert_allclose(
            pos_after[axis_idx], pos_before[axis_idx], atol=1e-5,
            err_msg="Axis-start atom (C15) should not move",
        )
        np.testing.assert_allclose(
            pos_after[pivot_idx], pos_before[pivot_idx], atol=1e-5,
            err_msg="Pivot atom (C16) should not move",
        )

        # All mobile atoms must have moved
        for idx in mobile:
            self.assertFalse(
                np.allclose(pos_after[idx], pos_before[idx], atol=1e-5),
                f"Mobile atom {idx} should have moved after 180° rotation",
            )

        # A second 180° must restore the mobile atoms
        self.flipmc.rotate_terminal(0)
        pos_restored = self._get_positions()
        np.testing.assert_allclose(
            pos_restored[mobile], pos_before[mobile], atol=1e-5,
            err_msg="Two 180° rotations should restore all mobile atom positions",
        )
        print("  PASSED: axis/pivot unmoved, mobile atoms rotated, double-180° restores positions")

        self.flipmc.move_dihe()
        self.flipmc.move_dihe()


if __name__ == "__main__":
    unittest.main()
