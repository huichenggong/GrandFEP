#!/usr/bin/env bash

# make sure argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target_part>"
    echo ""
    exit 1
fi
copydir=$1


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
    for rep in {0..2}
    do
        mkdir -p $base/$edge_name/$ff/$leg/$prot/rep_$rep
        cd       $base/$edge_name/$ff/$leg/$prot/rep_$rep
        pwd
        for win in {0..15}
        do
            mkdir -p $base/$edge_name/$ff/$leg/$prot/rep_$rep/$win/0
            cd       $base/$edge_name/$ff/$leg/$prot/rep_$rep/$win/0
            for fname in gc.rst7 gc.jsonl gc.log
            do
                # if there is no such file or link, and there is a file under ../../../../XXXX/rep_$rep/$win/0/, link it
                if [ ! -e $fname ] && [ -e ../../../../$copydir/rep_$rep/$win/0/$fname ]; then
                    ln -s ../../../../$copydir/rep_$rep/$win/0/$fname $fname
                    echo "Linked ../../../../$copydir/rep_$rep/$win/0/$fname"
                fi
            done
        done
    done
done < <($script_dir/csv_emit.py edge.csv edge_name)
