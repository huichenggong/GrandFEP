#!/usr/bin/env bash
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev

# make sure meta.sh and memory.sh exist
for filename in meta.sh memory.sh; do
    if [[ ! -f $filename ]]; then
        echo "$filename not found!"
        exit 1
    fi
done

source meta.sh
PR_cycle=${meta_data[PR_cycle]}


sys_name=$(basename $(dirname $(dirname $(dirname $(dirname $PWD)))))
ff=$(basename $(dirname $(dirname $PWD)))
leg=$(basename $(dirname $PWD))
prot=$(basename $PWD)

echo "System     : $sys_name"
echo "water ff   : $ff"
echo "Alchem leg : $leg"
echo "Protocol   : $prot"
echo "PR cycle   : $PR_cycle"

cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name lig1 lig2; do
    for rep in 0 1 2 10
    do
        if [ ! -d $base/$edge_name/$ff/$leg/$prot/rep_$rep/ ]; then
            echo "$edge_name rep_$rep not found!"
            continue
        fi
        cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
        pwd

        part_prev=1
        for part in {2..30}
        do
            if [ ! -d $base/$edge_name/$ff/$leg/$prot/rep_$rep/0/$part ]; then
                continue
            fi
            cd $base/$edge_name/$ff/$leg/$prot/rep_$rep

            log_array=()
            rst_now_array=()
            rst_prev_array=()
            for win in {0..15}
            do
                log_array+=("$win/$part/gc.log")
                rst_now_array+=("$win/$part/gc.rst7")
                rst_prev_array+=("$win/$part_prev/gc.rst7")
            done


            # all of the rst_now and log must exist. At least one rst_prev must exist
            all_done=true
            any_exist=false
            for win in {0..15}
            do
                if [ ! -f "${log_array[$win]}" ] || [ ! -f "${rst_now_array[$win]}" ] ; then
                    all_done=false
                    break # File missing, short circuit
                elif ! grep -q "RE Step $((PR_cycle * 5 -1))" "${log_array[$win]}" ; then
                    all_done=false
                    break # Simulation not finished, short circuit
                fi
                if [ -f "${rst_prev_array[$win]}" ] ; then
                    any_exist=true
                fi
            done
            if [ "$all_done" = true ] && [ "$any_exist" = true ] ; then
                printf "clean rep_%d part %2d\n" "$rep" "$part_prev"
                for win in {0..15}
                do
                    rm -f "${rst_prev_array[$win]}"
                done
            fi
            part_prev=$part

        done


    done
done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
