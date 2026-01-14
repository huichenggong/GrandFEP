#!/usr/bin/env bash
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev

sys_name=$(basename $(dirname $(dirname $(dirname $(dirname $PWD)))))
ff=$(basename $(dirname $(dirname $PWD)))
leg=$(basename $(dirname $PWD))
optrest=$(basename $PWD)

echo "water ff   : $ff"
echo "Alchem leg : $leg"
echo "Opt rest   : $optrest"

cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name lig1 lig2; do
    for rep in 0 1 2
    do
        cd $base/$edge_name/$ff/$leg/$optrest/rep_$rep
        pwd

        sed "s/JOBNAME/${sys_name}_${ff}_${leg}_${rep}_${edge_name}/g" $base/JOB/$ff/$leg/$optrest/rep_999/job_02_prod.sh > job_02_prod.sh
        if [[ $(hostname) == "owl5" || $(hostname) == "owl6"  ]]; then
            sed -i "s/ EXCLUDE/SBATCH --exclude=n11-13,n14-[01-11,13-18,20,22-32,34-35,40],n21-[01-07,09-20,23-42],n27-30,n19-10,n13-24,n18-12,n26-[02,04,06,08,10,12,14,16,18,20,22,24],n28-[02,04,06,08,10,12,14,16,18,20,22,24],n38-45,n39-[02,04,06,08,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46]/g" \
            job_02_prod.sh
        elif [[ $(hostname) == "moa1" || $(hostname) == "moa2" || $(hostname) == "moa" ]]; then
            sed -i "s/ EXCLUDE/SBATCH --exclude=n18-[12,21],,n26-[04,06,08,10,14,16,18,20,22,24],n39-[02,04,06,08,10,12,14,16,18,20,22,24,26,28,30,32,34,36,40,42,44,46]/g" \
            job_02_prod.sh
        fi
        sbatch job_02_prod.sh
    done
done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
