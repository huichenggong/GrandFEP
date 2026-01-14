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
    pwd
    # grep "INFO: Box_" gc_start.log
    # grep "INFO: rep_" gc_start.log | cat -n | tail -n 1
    for rep in {0..2}
    do
        cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
        pwd
        for win in {0..15}
        do
            cd $base/$edge_name/$ff/$leg/$prot/rep_$rep/$win
            # rename the gc_start?.rst7 to gc_start.rst7
            for f in gc_start?.rst7; do
                mv "$f" "gc_start.rst7"
            done
            for f in gc_start?.jsonl; do
                mv "$f" "gc_start.jsonl"
            done
        done
    done
done < <($script_dir/csv_emit.py edge.csv edge_name)
