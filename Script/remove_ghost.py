#!/usr/bin/env python


import sys
import argparse
import json

import numpy as np
from tqdm import tqdm

import MDAnalysis as mda
import MDAnalysis.transformations as trans

import grandfep
from grandfep import utils

from openmm import app


def read_ghosts_from_jsonl(file_path):
    ghost_list = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if isinstance(data, dict) and 'ghost_list' in data and 'dcd' in data and data['dcd']==1:
                ghost_list.append(data['ghost_list'])
    return ghost_list

def main():
    parser = argparse.ArgumentParser(description=
                 f"""GrandFEP {grandfep.__version__}. This script can shift the ghost water out side the simulation box, and
                     save the processed trajectory to a xtc. Automatic centering is still under development.""", )
    parser.add_argument("-p", metavar="       top.psf/top.parm7/top.pdb", default="top.pdb",
                        help="Input topology file")
    parser.add_argument("-idcd", metavar="    mdX.dcd", nargs='+',
                        help="Input dcd files")
    parser.add_argument("-ijsonl", metavar="  mdX.jsonl", nargs='+',
                        help="Input ghosts.jsonl files")
    parser.add_argument("-oxtc", metavar="    md.xtc", default="md.xtc",
                        help="Output xtc files, the ghost waters will be shifted outside the simulation cell.")
    args = parser.parse_args()

    print(f"GrandFEP Version: {grandfep.__version__}")
    print(f"Command: {" ".join(sys.argv)}")
    
    if args.p.endswith(".pdb"):
        pdb = app.PDBFile(args.p)
        top = pdb.topology
    elif args.p.endswith(".psf") or args.p.endswith(".parm7"):
        top, _ = utils.load_top(args.p)
    else:
        raise ValueError(f"Unsupported topology file format: {args.p}. Supported formats are .pdb, .psf, and .parm7.")

    ghost_list = []
    for ghost_f, dcd in zip(args.ijsonl, args.idcd):
        print(f"Load ghost {ghost_f}, and trajectory {dcd}")
        ghosts = read_ghosts_from_jsonl(ghost_f)
        ghost_list.extend(ghosts)

    u = mda.Universe(top, args.idcd)
    print(f"The number of frames in the trajectory: {len(u.trajectory)}")
    print(f"The number of frames in the ghost_list: {len(ghost_list)}")

    not_protein = u.select_atoms('not protein')
    protein = u.select_atoms('protein')
    all_atoms = u.select_atoms("name *")

    res_list = list(top.residues())
    with mda.Writer(args.oxtc, u.atoms.n_atoms) as xtc_writer:
        for ts, ghosts in tqdm(zip(u.trajectory, ghost_list)):
            z = -20
            for resid in ghosts:
                res = res_list[resid]
                at_list = list(res.atoms())
                shift_array = np.array([-5, -5, z*2]) - ts.positions[at_list[0].index]
                for atom in res.atoms():
                    ts.positions[atom.index] += shift_array
                z += 1
            xtc_writer.write(u.atoms)


if __name__ == "__main__":
    main()
