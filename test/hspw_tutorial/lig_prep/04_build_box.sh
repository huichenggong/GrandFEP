#!/usr/bin/env bash

cd ../
base=$PWD

lig_count=1

# loop through edge.csv (ignore first line), and split each line into variables
while IFS=$'\x1f' read -r lig1 expddg; do
    cd $base/lig_prep/$lig1
    index_name=M$(printf "%02d" $lig_count)
    echo $index_name
    grep $index_name ../03_all_ligands_solvated.pdb >  06_box.pdb
    sed -i "s/$index_name/MOL/g" 06_box.pdb
    grep WAT         ../03_all_ligands_solvated.pdb >> 06_box.pdb
    tleap -f ../04_tip3p_150mmol.in
    tleap -f ../05_OPC.in


    lig_count=$((lig_count+1))
done < <($script_dir/csv_emit.py lig.csv Lig_Name Exp_dG)
