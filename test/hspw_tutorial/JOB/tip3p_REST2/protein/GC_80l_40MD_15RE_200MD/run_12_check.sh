#!/usr/bin/env bash
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev

# make sure meta.sh exist
if [[ ! -f meta.sh ]]; then
    echo "meta.sh not found!"
    exit 1
fi

replicas=${1:-"0 1 2 10"}

source meta.sh

sys_name=$(basename $(dirname $(dirname $(dirname $(dirname $PWD)))))
ff=$(basename $(dirname $(dirname $PWD)))
leg=$(basename $(dirname $PWD))
prot=$(basename $PWD)
PR_cycle=${meta_data[PR_cycle]}

echo "System     : $sys_name"
echo "water ff   : $ff"
echo "Alchem leg : $leg"
echo "Replicas   : $replicas"
echo "Protocol   : $prot"
echo "PR cycle   : $PR_cycle"

cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name lig1 lig2; do
    if [ ! -d $base/$edge_name/$ff/$leg/$prot/ ]; then
        continue
    fi
    cd $base/$edge_name/$ff/$leg/$prot/
    pwd
    echo "       123456789012345678901234567890"
    for rep in $replicas
    do
        if [ ! -d $base/$edge_name/$ff/$leg/$prot/rep_$rep ]; then
            continue
        fi
        cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
        n_finished=0
        for part in {1..100}
        do
            if [ -f 2/${part}/gc.log ] ; then
                restep=$(tac 2/${part}/gc.log | grep "RE Step " -m1 )
                if [[ $restep == *"RE Step $((PR_cycle * 5 - 1))"* ]]; then
                    ((n_finished++))
                fi
            else
                break
            fi
        done
        njob_queue=$(squeue -u chui --format="%.8i %.2t %Z MARK" | grep -v " CG " | grep "$PWD MARK"  |  wc -l)
        n_total=$(($njob_queue + $n_finished))
        n_remaining=$((30 - $n_total))
        progress_bar=""
        for ((i=0; i<n_finished; i++)); do
            progress_bar+="X"
        done
        for ((i=0; i<njob_queue; i++)); do
            progress_bar+="-"
        done
        for ((i=0; i<n_remaining; i++)); do
            progress_bar+="_"
        done
        printf "rep_%-2d %s %2d / %2d\n" "$rep" "$progress_bar" "$n_finished" "$n_total"
    done

done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
