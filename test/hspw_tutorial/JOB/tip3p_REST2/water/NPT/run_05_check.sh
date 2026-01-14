#!/usr/bin/env bash
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev

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
        cd $base/$edge_name/$ff/$leg/$optrest/rep_$rep
        pwd
        for part in 0 1
        do
            if [ ! -f 2/${part}/npt.log ] ; then
                echo "ERROR: no npt.log"
                continue
            fi

            restep=$(tac 2/${part}/npt.log | grep "RE Step " -m1 )
            if [[ $restep == *"RE Step 9999"* ]]; then
                :
            else
                echo $restep
            fi

        done
    done

    
done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
