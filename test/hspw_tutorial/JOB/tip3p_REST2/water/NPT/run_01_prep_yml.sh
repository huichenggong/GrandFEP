#!/usr/bin/env bash

sys_name=$(basename $(dirname $(dirname $(dirname $(dirname $PWD)))))
ff=$(basename $(dirname $(dirname $PWD)))
leg=$(basename $(dirname $PWD))
optrest=$(basename $PWD)

echo "water ff   : $ff"
echo "Alchem leg : $leg"
echo "Opt rest   : $optrest"

cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name lig1 lig2; do
    mkdir -p $base/$edge_name/$ff/$leg/$optrest/rep_999
    cd       $base/$edge_name/$ff/$leg/$optrest/rep_999
    pwd
    # sed keyword GEN_VEL from npt_tmp.yml to npt_eq.yml
    sed 's/GEN_VEL/true/'  ../../../../../JOB/$ff/$leg/$optrest/rep_999/npt_tmp.yml > ./npt_eq.yml
    sed 's/GEN_VEL/false/' ../../../../../JOB/$ff/$leg/$optrest/rep_999/npt_tmp.yml > ./npt.yml

    for win in {0..15}
    do
        mkdir $win -p
        sed "s/INIT_LAMBDA_STATE/$win/" npt_eq.yml > $win/npt_eq.yml
        sed "s/INIT_LAMBDA_STATE/$win/" npt.yml    > $win/npt.yml
        cat ../../../OPT_lam.yml >> $win/npt_eq.yml
        cat ../../../OPT_lam.yml >> $win/npt.yml
    done

    
done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
