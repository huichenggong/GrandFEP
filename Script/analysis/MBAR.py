#!/usr/bin/env python

import sys
from pathlib import Path
import argparse

import pandas as pd

from openmm import unit
from grandfep import utils




parser = argparse.ArgumentParser()
parser.add_argument("-log", type=Path, required=True, nargs='+',
                    help="log files")
parser.add_argument("-kw",  type=str, default="Reduced Energy U_i(x):",
                    help="Keyword to parse the log files")
parser.add_argument("-sep", type=str, default=",",
                    help="String that separates the energy values in the log files")
parser.add_argument("-b", "--begin",   type=int, default=0,
                    help="First frame to analyze")
parser.add_argument("-t", "--temperature", type=float,
                    help="Temperature in Kelvin")
parser.add_argument("-csv", type=Path, default=None,
                    help="Output CSV file to save the results. If not provided, results will not be saved to a csv file.")
parser.add_argument(
    "--no-drop-eq",          # long option, clearer name
    dest="drop_eq",          # variable will still be called drop_eq
    action="store_false",
    help="Do NOT drop the equilibration frames (default: drop them)")
parser.add_argument("-m", "--method", default=["MBAR", "BAR"], nargs="+",
                    choices=["MBAR", "BAR"],
                    help="Method to use for free energy calculation (MBAR or BAR)")

args = parser.parse_args()


print(f"Command line arguments: {' '.join(sys.argv)}")

print(f"{args.drop_eq =}")

analysis = utils.FreeEAnalysis(args.log, args.kw, args.sep, drop_equil=args.drop_eq, begin=args.begin)

if args.temperature is not None:
    print(f"Set Temperature to {args.temperature} K as the given argument.")
    analysis.set_temperature(args.temperature * unit.kelvin)
print()
analysis.print_uncorrelate()

res_all = {}
for method in args.method:
    print(f"Calculating free energy using {method} method.")
    if method == "MBAR":
        res_all[method] = analysis.mbar_U_all()
    elif method == "BAR":
        res_all[method] = analysis.bar_U_all()

analysis.print_res_all(res_all)

if args.csv is not None:
    res_dict = {}
    for k, (dG, dG_err, v) in res_all.items():
        # dG is a NxN matrix. It the freeE difference between the states.
        for i, dGi in enumerate(dG):
            res_dict[f"{k}_{i}"] = dGi
            res_dict[f"{k}_{i}_err"] = dG_err[i]
    df = pd.DataFrame(res_dict)
    df.to_csv(args.csv, index=False)

