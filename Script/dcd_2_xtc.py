#!/usr/bin/env python
# -*- coding: utf-8 -*-

# convert dcd files to xtc files using mdanalysis
import argparse

import MDAnalysis as mda

def convert_dcd_to_xtc(pdb_file, dcd_file, xtc_file):
    u = mda.Universe(pdb_file, dcd_file)
    with mda.Writer(xtc_file, n_atoms=u.atoms.n_atoms) as W:
        for ts in u.trajectory:
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

