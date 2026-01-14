#!/usr/bin/env bash
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev

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

while IFS=$'\x1f' read -r edge_name ; do
    # for rep in 0 1 2
    # do
    #     cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
    #     pwd
    #     $script_dir/check_RE.sh 0/0/gc.log
    # done
    cd $base/$edge_name/$ff/$leg/$prot/
    pwd
    $script_dir/analysis/check_RE.sh rep_?/0/0/gc.log
    echo ""

done < <($script_dir/csv_emit.py edge.csv edge_name)
