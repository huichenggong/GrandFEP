#!/usr/bin/env python3
import sys, argparse
import pandas as pd

US = "\x1f"  # ASCII Unit Separator (unlikely to appear in data)

ap = argparse.ArgumentParser(description="Emit selected CSV columns for bash")
ap.add_argument("csv", help="input CSV file")
ap.add_argument("cols", nargs="+", help="column names to emit (in order)")
ap.add_argument("--skip-rows", type=int, default=0, help="skip first N rows")
ap.add_argument("--na", default="", help="string to use for NA values")
ap.add_argument("--delim", default=US, help="field delimiter (default: unit separator)")
args = ap.parse_args()

df = pd.read_csv(args.csv)
if args.skip_rows:
    df = df.iloc[args.skip_rows:]

for row in df[args.cols].itertuples(index=False):
    fields = []
    for v in row:
        s = args.na if pd.isna(v) else str(v)
        # keep it single-line and delimiter-safe
        s = s.replace("\n", " ").replace("\r", " ").replace(args.delim, " ")
        fields.append(s)
    sys.stdout.write(args.delim.join(fields) + "\n")
