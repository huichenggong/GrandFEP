#!/usr/bin/env python
# -*- coding: utf-8 -*-

# convert dcd files to xtc files using mdanalysis
import argparse

import MDAnalysis as mda
from MDAnalysis.lib.distances import apply_PBC


def _standardize_box(ts):
    """
    Standardize the periodic box to use beta > 90° (cos(beta) < 0).

    For rhombic dodecahedra two equivalent unit cells exist that differ
    only in the sign of the v3 off-diagonal components:
      standard : beta ≈ 120°, cos(beta) ≈ -0.5
      flipped  : beta ≈  60°, cos(beta) ≈ +0.5

    When different lambda windows are initialised from restart files that
    use different conventions, a replica-exchange simulation that swaps
    coordinates and box vectors will mix frames from both conventions in
    the same trajectory.  The result is that beta appears to jump between
    60° and 120° across frames, which VMD interprets as a change in cell
    shape rather than a change in cell size.

    This function converts any beta < 90° frame to the beta > 90°
    convention in-place and re-wraps atom positions into the new unit
    cell via fractional coordinate mapping.
    """
    if ts.dimensions[4] < 90.0:
        ts.dimensions[4] = 180.0 - ts.dimensions[4]
        ts.positions[:] = apply_PBC(ts.positions, ts.dimensions)


def convert_dcd_to_xtc(pdb_file, dcd_file, xtc_file):
    u = mda.Universe(pdb_file, dcd_file)
    with mda.Writer(xtc_file, n_atoms=u.atoms.n_atoms) as W:
        for ts in u.trajectory:
            _standardize_box(ts)
            W.write(u)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCD to XTC")
    parser.add_argument("-pdb", help="Input pdb file")
    parser.add_argument("-dcd", help="Input DCD files", nargs='+')
    parser.add_argument("-xtc", help="Output XTC file")
    args = parser.parse_args()

    print(f"pdb : {args.pdb}")
    print(f"dcd : {args.dcd}")
    print(f"xtc : {args.xtc}")

    convert_dcd_to_xtc(args.pdb, args.dcd, args.xtc)

