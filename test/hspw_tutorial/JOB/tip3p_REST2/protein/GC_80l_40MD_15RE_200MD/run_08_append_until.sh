#!/usr/bin/env bash

# make sure argument is provided
if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: $0 <target_part> <rep>"
    echo ""
    exit 1
fi
target=$1
# if rep is not provided, default to "0 1 2 10"
replicas=${2:-"0 1 2 10"}

script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev

# make sure meta.sh and memory.sh exist
for filename in meta.sh memory.sh; do
    if [[ ! -f $filename ]]; then
        echo "$filename not found!"
        exit 1
    fi
done

source memory.sh
source meta.sh

sys_name=$(basename $(dirname $(dirname $(dirname $(dirname $PWD)))))
ff=$(basename $(dirname $(dirname $PWD)))
leg=$(basename $(dirname $PWD))
prot=$(basename $PWD)

EQ_cycle=${meta_data[EQ_cycle]}
PR_cycle=${meta_data[PR_cycle]}

memMB=${mem_map[$sys_name]}
maxh=${meta_data["PR_maxh_$sys_name"]}
ngpu=${meta_data["ngpu_$sys_name"]}

echo "System     : $sys_name"
echo "water ff   : $ff"
echo "Alchem leg : $leg"
echo "Protocol   : $prot"
echo "Memory     : $memMB"
echo "EQ cycle   : $EQ_cycle"
echo "PR cycle   : $PR_cycle"
echo "Max hours  : $maxh"
echo "GPUs       : $ngpu"
echo "Target jobs: $target"
echo "Replicas   : $replicas"

cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name lig1 lig2; do
    for rep in $replicas
    do
        if [ ! -d $base/$edge_name/$ff/$leg/$prot/rep_$rep ] ; then
            echo "$base/$edge_name/$ff/$leg/$prot/rep_$rep"
            echo "Dir does not exist, skipping"
            continue
        fi
        cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
        pwd

        njob_queue=$(squeue -u chui --format="%.8i %.2t %Z MARK" | grep -v " CG " | grep "$PWD MARK"  |  wc -l)
        n_finished=0
        for part in {0..100}
        do
            if [ -f 2/${part}/gc.log ] ; then
                restep=$(tac 2/${part}/gc.log | grep "RE Step " -m1 )
                if [[ $restep == *"RE Step $((PR_cycle * 5 -1))"* ]]; then
                    ((n_finished++))
                fi
            else
                break
            fi
        done
        target_njob=$(($target - $njob_queue - $n_finished))
        echo "Submit $target_njob jobs"
        for part in $(seq 1 $target_njob)
        do
            echo "$part submit"
            sbatch job_03_append.sh
        done
    done
done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
