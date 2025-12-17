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
    for rep in 0 1 2
    do
        cd $base/$edge_name/$ff/$leg/$optrest/
        cp -r rep_999 rep_$rep
        cd rep_$rep
        pwd
        sed "s/JOBNAME/${sys_name}_${ff}_${leg}_${rep}_${edge_name}/g" $base/JOB/$ff/$leg/$optrest/rep_999/job_01_eq.sh > job_01_eq.sh
        sbatch job_01_eq.sh
    done

    
done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
