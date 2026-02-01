#!/usr/bin/env bash
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev

sys_name=$(basename $(dirname $(dirname $(dirname $(dirname $PWD)))))
ff=$(basename $(dirname $(dirname $PWD)))
leg=$(basename $(dirname $PWD))
prot=$(basename $PWD)

echo "water ff   : $ff"
echo "Alchem leg : $leg"
echo "Protocol   : $prot"


cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name lig1 lig2; do
    for rep in 0 1 2
    do
        cd $base/$edge_name/$ff/$leg/$prot/
        if [ -f rep_$rep/15/npt_eq.pdb ]; then
            echo "Skipping $edge_name rep_$rep as npt_eq.pdb exists"
            continue
        fi
        
        cp -r rep_999 rep_$rep
        cd rep_$rep
        pwd

        sed "s/JOBNAME/${sys_name}_${ff}_${leg}_${rep}_${edge_name}/g" $base/JOB/$ff/$leg/$prot/rep_999/job_01_eq.sh > job_01_eq.sh
        # if [[ $(hostname) == "owl5" || $(hostname) == "owl6"  ]]; then
        #     # sed -i "s/ EXCLUDE/SBATCH --exclude=n11-13,n14-[01-11,13-18,20,22-32,34-35,40],n21-[01-07,09-20,23-42],n27-30,n19-10,n13-24,n18-12,n26-[02,04,06,08,10,12,14,16,18,20,22,24],n28-[02,04,06,08,10,12,14,16,18,20,22,24],n38-45,n39-[02,04,06,08,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46]/g" \
        #     # job_01_eq.sh
        #     echo
        # elif [[ $(hostname) == "moa1" || $(hostname) == "moa2" || $(hostname) == "moa" ]]; then
        #     # sed -i "s/ EXCLUDE/SBATCH --exclude=n50-[23-36],n75-[01-20,22-42],n18-[02,04,06,08,10,12,14,16,19,21,23,25,27,29,32,34,36,38,40,42],n32-[02,04,06,08,10,12,14,16,18,20,22,23-46],n26-[04,06,08,10,14,16,18,20,22,24,26-30],n39-[02,04,06,08,10,12,14,16,18,20,22,24,26,28,30,32,34,36,40,42,44,46,47]/g" \
        #     sed -i "s/ EXCLUDE/SBATCH --exclude=n18-[36,38]/g" \
        #     job_01_eq.sh
        # fi
        sbatch job_01_eq.sh
    done
    sleep 1

done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
