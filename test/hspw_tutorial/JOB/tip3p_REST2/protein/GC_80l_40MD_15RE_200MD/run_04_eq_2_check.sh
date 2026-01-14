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
        for win in 0
        do
            cd $base/$edge_name/$ff/$leg/$prot/rep_$rep/$win/
            # there should be 3 lines in gc.jsonl
            if [ -f 0/gc.jsonl ]; then
                line_count=$(grep -c '"dcd": 1' 0/gc.jsonl)
            else
                line_count=0
            fi
            # echo -n "rep_$rep $line_count "
            printf "rep_%-2s %2s " "$rep" "$line_count"
            # make a progress bar, each 'X' represent 1 line, 10% each
            for i in $(seq 1 $line_count)
            do
                echo -n "X"
            done
            # 10 - line_count space
            for i in $(seq 1 $((10 - line_count)))
            do
                echo -n " "
            done

            # '"RE_step": EQ_cycle x 3' should be present in gc.jsonl
            if [ -f 0/gc.jsonl ] && grep -q "\"RE_step\": $((EQ_cycle * 3))" 0/gc.jsonl; then
                echo "✓"
            else
                echo "✗"
            fi
        done
    done
done < <($script_dir/csv_emit.py edge.csv edge_name)
