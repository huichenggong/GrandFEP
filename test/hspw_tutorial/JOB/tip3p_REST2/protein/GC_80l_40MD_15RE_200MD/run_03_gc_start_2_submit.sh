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
        cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
        pwd
        any_missing=false
        for win in {0..15}
        do
            if [ ! -f $win/gc_start.rst7 ]; then
                any_missing=true
                break
            fi
        done
        if [ "$any_missing" = true ] ; then
            sed -e "s/JOBNAME/${sys_name}-$leg-$edge_name-$rep/" \
                ../../../../../JOB/$ff/$leg/$prot/rep_999/job_2_gc_start.sh > job_2_gc_start.sh
            if [[ $(hostname) == "owl5" || $(hostname) == "owl6"  ]]; then
                sed -i "s/ EXCLUDE/SBATCH --exclude=n28-26,n19-30,n14-35/g" \
                job_2_gc_start.sh
            elif [[ $(hostname) == "moa1" || $(hostname) == "moa2" || $(hostname) == "moa" ]]; then
                # sed -i "s/ EXCLUDE/SBATCH --exclude=n50-[23-36],n75-[01-20,22-42],n18-[02,04,06,08,10,12,14,16,19,21,23,25,27,29,32,34,36,38,40,42],n32-[02,04,06,08,10,12,14,16,18,20,22,23-46],n26-[04,06,08,10,14,16,18,20,22,24,26-30],n39-[02,04,06,08,10,12,14,16,18,20,22,24,26,28,30,32,34,36,40,42,44,46,47]/g" \
                # sed -i "s/ EXCLUDE/SBATCH --exclude=n18-[36,38]/g" \
                # job_2_gc_start.sh
                echo ""
            fi
            if [ ! -f 0/gc_start.rst7 ]; then
                sbatch job_2_gc_start.sh
            fi
        else
            echo "rep_$rep already done."
        fi
    done


    # break
done < <($script_dir/csv_emit.py edge.csv edge_name)
