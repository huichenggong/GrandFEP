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

cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name ; do
    for rep in 0 1 2 10
    do
        if [ ! -d $base/$edge_name/$ff/$leg/$prot/rep_$rep ]; then
            continue
        fi
        cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
        pwd

        sed  \
            -e "s/JOBNAME/${sys_name}_${ff}_${leg}_${rep}_${prot}_${edge_name}/" \
            -e "s/NCYCLE/$((PR_cycle * 5))/" \
            -e "s/0CYCLE/$((EQ_cycle * 3))/" \
            -e "s/MEMORY/$memMB/" \
            -e "s/MAXHSL/$((maxh + 1))/" \
            -e "s/MAXHMD/$maxh/" \
            -e "s/NGPU/$ngpu/" \
            $base/JOB/$ff/$leg/$prot/rep_999/job_03_append.sh > job_03_append.sh
        
        # exclude 1080, 1080ti, 2080 nodes
        if [[ $(hostname) == "owl5" || $(hostname) == "owl6"  ]]; then
            sed -i "s/ EXCLUDE/SBATCH --exclude=n14-[36,38-39,41],n11-13,n14-[01-11,13-18,20,22-32,34-35,40],n21-[01-07,09-20,23-42],n27-30,n13-24,n26-[02,04,06,08,10,12,14,16,18,20,22,24],n28-[02,04,06,08,10,12,14,16,18,20,22,24],n38-45,n39-[02,04,06,08,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46],n38-47/g" \
            job_03_append.sh
        elif [[ $(hostname) == "moa1" || $(hostname) == "moa2" || $(hostname) == "moa" ]]; then
            sed -i "s/ EXCLUDE/SBATCH --exclude=n26-[26-30],n32-[23-46],n50-[23-36],n75-[01-20,22-42],n18-[02,04,06,08,10,14,16,19,21,23,25,27,29,32,34,36,38,40,42],n32-[02,04,06,08,10,12,14,16,18,20,22],n39-47,n18-12,n26-[04,06,08,10,14,16,18,20,22,24],n39-[02,04,06,08,10,12,14,16,18,20,22,24,26,28,30,32,34,36,40,42,44,46]/g" \
            job_03_append.sh
        fi
        # sbatch job_03_append.sh
    done
done < <($script_dir/csv_emit.py edge.csv edge_name)
