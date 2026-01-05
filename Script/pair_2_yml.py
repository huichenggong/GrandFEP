#!/usr/bin/env python3

from pathlib import Path
import argparse

import yaml
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-dat", type=str, default="pairs_checked.dat", help="pmx pairs dat file")
parser.add_argument("-yml", type=str, default="map.yml", help="output yml file")

args = parser.parse_args()


#  _, lig1, lig2 = edge.name.split('_')
pair = np.loadtxt(args.dat, dtype=int)
pair_dict = {i.item()-1:j.item()-1 for i,j in pair}
yml_dict = {
    "mapping_list":[
        {
            "res_nameA": "MOL",
            "res_nameB": "MOL",
            "index_map":pair_dict
        },
    ]}
yml_file = Path(args.yml)
with open(yml_file, 'w') as f:
    yaml.dump(yml_dict, f)

