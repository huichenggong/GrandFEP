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
    cd       $base/$edge_name/$ff/$leg/$optrest/
    pwd
    for rep in 0 1 2
    do
        
        if [ -d $base/$edge_name/$ff/$leg/$optrest/rep_$rep ]; then
            cd $base/$edge_name/$ff/$leg/$optrest/rep_$rep
            echo rep_$rep
        else
            echo "rep_$rep does not exist. Skipping."
            continue
        fi
        for win in {0..15}
        do
            if [ ! -f $win/npt_eq.pdb ]; then
                if [ ! -f $win/npt_eq.csv ]; then
                    echo "$win/npt_eq.csv does not exist."
                else
                    # if non-empty file, tail the last line
                    if [ -s $win/npt_eq.csv ]; then
                        tail $win/npt_eq.csv -n 1
                    else
                        echo "$win/npt_eq.csv is empty."
                    fi
                fi
            fi
        done
    done

    
done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
