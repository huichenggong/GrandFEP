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
    for rep in {0..2}
    do
        mkdir -p $base/$edge_name/$ff/$leg/$prot/rep_$rep
        cd       $base/$edge_name/$ff/$leg/$prot/rep_$rep
        pwd
        for win in {0..15}
        do
            mkdir -p $base/$edge_name/$ff/$leg/$prot/rep_$rep/$win
            cd       $base/$edge_name/$ff/$leg/$prot/rep_$rep/$win
            for fname in gc_start.rst7 gc_start.jsonl
            do
                # if there is no such file or link, and there is a file under ../../../XXXXXX/rep_$rep/$win/, link it
                copydir="GC_80l_40MD_31RE_200MD"
                if [ ! -e $fname ] && [ -e ../../../$copydir/rep_$rep/$win/$fname ]; then
                    ln -s ../../../$copydir/rep_$rep/$win/$fname $fname
                    echo "Linked ../../../$copydir/rep_$rep/$win/$fname"
                fi
            done
        done
    done


    # break
done < <($script_dir/csv_emit.py edge.csv edge_name)
