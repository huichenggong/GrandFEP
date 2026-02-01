#!/usr/bin/env bash

script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev

source memory.sh
source meta.sh

sys_name=$(basename $(dirname $(dirname $(dirname $(dirname $PWD)))))
ff=$(basename $(dirname $(dirname $PWD)))
leg=$(basename $(dirname $PWD))
prot=$(basename $PWD)

EQ_cycle=${meta_data[EQ_cycle]}
PR_cycle=${meta_data[PR_cycle]}

memMB=${mem_map[$sys_name]}
maxh=${meta_data["EQ_maxh_$sys_name"]}

echo "System     : $sys_name"
echo "water ff   : $ff"
echo "Alchem leg : $leg"
echo "Protocol   : $prot"
echo "Memory     : $memMB"
echo "EQ cycle   : $EQ_cycle"
echo "PR cycle   : $PR_cycle"
echo "Max hours  : $maxh"

cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name ; do
    cd $base/$edge_name/$ff/$leg/$prot/
    pwd
    for rep in 0 1 2 10
    do
        if [ ! -d $base/$edge_name/$ff/$leg/$prot/rep_$rep ]; then
            continue
        fi
        cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
        for win in 2
        do
            cd $base/$edge_name/$ff/$leg/$prot/rep_$rep/$win/
            # '"RE_step": EQ_cycle x 3 - 1' should be present in npt.log
            if [ -f 0/npt.log ] && [ -f 0/npt.rst7 ] && grep -q "RE Step $((EQ_cycle * 3 - 1 ))" 0/npt.log; then
                echo "✓"
            else
                echo "rep_$rep ✗"
            fi
        done
    done
done < <($script_dir/csv_emit.py edge.csv edge_name)
