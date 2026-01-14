#!/usr/bin/env bash

source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"

sys_name=$(basename $(dirname $(dirname $(dirname $(dirname $PWD)))))
ff=$(basename $(dirname $(dirname $PWD)))
leg=$(basename $(dirname $PWD))
prot=$(basename $PWD)

echo "System     : $sys_name"
echo "water ff   : $ff"
echo "Alchem leg : $leg"
echo "Protocol   : $prot"

cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name; do
    if [ ! -d $base/$edge_name/$ff/$leg/$prot/rep_999 ]; then
        continue
    fi
    cd       $base/$edge_name/$ff/$leg/$prot/
    echo "$PWD"
    for rep in 0 1 2 10
    do
        cd       $base/$edge_name/$ff/$leg/$prot/rep_$rep
        for win in {0..15}
        do
            if [ ! -f $win/gc_start.rst7 ] && [ ! -f $win/0/gc.rst7 ] && [ ! -L $win/0/gc.rst7 ] ; then
                echo "rep_$rep Missing $win"
                break
            fi
        done
    done


    # break
done < <($script_dir/csv_emit.py edge.csv edge_name)
