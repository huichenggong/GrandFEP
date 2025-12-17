#!/usr/bin/env bash
cd ../../
base=$PWD

# loop through edge.csv (ignore first line), and split each line into variables
while IFS=$'\x1f' read -r edge_name lig1 lig2; do
    echo $edge_name $lig1 $lig2
    cd $base/$edge_name/
    pwd
    pymol -d "run $script_dir/color_map_pairs.py; check_color_mapping ../lig_prep/$lig1/01.pdb, ../lig_prep/$lig2/01.pdb, pairs_checked.dat" </dev/null
done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
