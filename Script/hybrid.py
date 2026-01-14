#!/usr/bin/env python

from pathlib import Path
import argparse
import sys
import time
import gzip

import numpy as np
from mpi4py import MPI

from openmm import app, unit, openmm

from grandfep import utils, sampler

def main():
    time_start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("-prmtopA",  type=str, required=True,
                        help="Amber prmtop file for state A")
    parser.add_argument("-prmtopB",  type=str, required=True,
                        help="Amber prmtop file for state B")
    parser.add_argument("-inpcrdA",  type=str, required=True,
                        help="Amber inpcrd/rst7 file for state A")
    parser.add_argument("-inpcrdB",  type=str, required=True,
                        help="Amber inpcrd/rst7 file for state B")
    parser.add_argument("-yml" ,     type=str, required=True, 
                        help="Yaml md parameter file. Atom mapping and nonbonded settings.")

    parser.add_argument("-pdb", type=str, required=True,
                        help="Output PDB file for the topology")
    parser.add_argument("-system", type=str, required=True,
                        help="Output Serialized system, can be .xml or .xml.gz")
    # if given, use REST2
    parser.add_argument("-REST2", action='store_true')
    parser.add_argument("-dum_dihe_scale", nargs="+", type=float, default=[0.0, 1.0, 0.0, 0.0, 0.0],
                        help="Dihedral scaling for dummy atoms. ")
    parser.add_argument("-REST2_res", type=str,
                        nargs="+")

    msg_list = []
    args = parser.parse_args()
    msg = f"Command line arguments: {' '.join(sys.argv)}"
    print(msg)
    msg_list.append(msg)

    mdp = utils.md_params_yml(args.yml)

    # Load the state A/B
    inpcrdA = app.AmberInpcrdFile(args.inpcrdA)
    inpcrdB = app.AmberInpcrdFile(args.inpcrdB)
    prmtopA = app.AmberPrmtopFile(args.prmtopA, periodicBoxVectors=inpcrdA.boxVectors)
    prmtopB = app.AmberPrmtopFile(args.prmtopB, periodicBoxVectors=inpcrdB.boxVectors)
    nonbonded = mdp.get_system_setting()

    sysA = prmtopA.createSystem(**nonbonded)
    sysB = prmtopB.createSystem(**nonbonded)

    # Hybrid A and B
    old_to_new_atom_map, old_to_new_core_atom_map = utils.prepare_atom_map(prmtopA.topology, prmtopB.topology, mdp.mapping_list)
    if args.REST2:
        print("Using REST2")
        scale_dihe = {
                1: 0.0, "i1": 1.0,
                2: 1.0, "i2": 1.0,
                3: 0.0, "i3": 1.0,
                4: 0.0, "i4": 1.0,
                5: 0.0, "i5": 1.0,
            }
        for i, scale in enumerate(args.dum_dihe_scale):
            print(f"Dummy dihedral scale for periodicity {i+1}: {scale}")
            scale_dihe[i+1] = scale
        if args.REST2_res is not None:
            old_rest2_atom_indices = []
            for res in args.REST2_res:
                print(f"Add REST2 residue: {res}")
                name, index_str = res.split(":")
                index = int(index_str)
                ind_list = utils.find_reference_atom_indices(prmtopA.topology, [{"res_name": name, "res_index": index}])
                old_rest2_atom_indices.extend(ind_list)
            print(f"REST2 residues specified: {args.REST2_res}, total atoms: {len(old_rest2_atom_indices)}")

        h_factory = utils.HybridTopologyFactoryREST2(
            sysA, inpcrdA.getPositions(), prmtopA.topology,
            sysB, inpcrdB.getPositions(), prmtopB.topology,
            old_to_new_atom_map,      # All atoms that should map from A to B
            old_to_new_core_atom_map, # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            scale_dihe=scale_dihe,
            old_rest2_atom_indices=old_rest2_atom_indices if args.REST2_res is not None else None)
        
        for dihe_key in ["old_only", "new_only"]:
            # for t_dict in [h_factory.hybrid_torsion_dict["old_only"], h_factory.hybrid_torsion_dict["new_only"]]:
            print(f"Torsions for {dihe_key}:")
            t_dict = h_factory.hybrid_torsion_dict[dihe_key]
            for torsion_key, param in t_dict.items():
                idx1, idx2, idx3, idx4, periodicity = torsion_key
                periodicity, phase, k, torsion_type, t_string = param
                print(f" {idx1+1:5d} {idx2+1:5d} {idx3+1:5d} {idx4+1:5d} {torsion_type:8s} {periodicity}", phase, k, t_string)
        
        dihe_key = "intersection"
        print(f"Torsions for intersection with different force constants:")
        for torsion_key, param in h_factory.hybrid_torsion_dict[dihe_key].items():
            idx1, idx2, idx3, idx4, periodicity = torsion_key
            periodicity_old, phase_old, k_old, torsion_type_old, \
                periodicity_new, phase_new, k_new, torsion_type_new, t_string = param
            if k_old != k_new:
                print(f" {idx1+1:5d} {idx2+1:5d} {idx3+1:5d} {idx4+1:5d} {torsion_type_old:8s}", phase_old, k_old, k_new, t_string)


    else:
        print("No REST2")
        h_factory = utils.HybridTopologyFactory(
            sysA, inpcrdA.getPositions(), prmtopA.topology,
            sysB, inpcrdB.getPositions(), prmtopB.topology,
            old_to_new_atom_map,      # All atoms that should map from A to B
            old_to_new_core_atom_map, # Alchemical Atoms that should map from A to B
            use_dispersion_correction=True,
            softcore_LJ_v2=False)
        
    
    
    
    system    = h_factory.hybrid_system
    topology  = h_factory.omm_hybrid_topology

    topology.setPeriodicBoxVectors(inpcrdA.boxVectors)
    system.setDefaultPeriodicBoxVectors(*inpcrdA.boxVectors)
    positions = h_factory.hybrid_positions

    with gzip.open(args.system, mode="wt") as fh:
        fh.write(openmm.XmlSerializer.serialize(system))

    app.PDBFile.writeFile(topology, positions, open(args.pdb, "w"))


    

if __name__ == "__main__":
    main()

