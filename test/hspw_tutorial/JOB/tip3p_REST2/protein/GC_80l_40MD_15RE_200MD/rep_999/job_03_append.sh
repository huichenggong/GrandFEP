#!/bin/bash
#
#SBATCH --job-name=JOBNAME
#SBATCH --partition=p16,p32,p40,p20,p24
#SBATCH --get-user-env
#SBATCH --gres=gpu:NGPU              # number of GPUs requested
#SBATCH -t MAXHSL:00:00                # hours:min:sec
#SBATCH --mem=MEMORY
#SBATCH --dependency=singleton
# EXCLUDE

source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
module add openmpi/gcc/64/4.1.5

echo ""
echo "$(date "+%Y-%m-%d_%H:%M:%S") Check Slurm Env"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

echo $(date "+%Y-%m-%d_%H:%M:%S") "Checking software Env on node $(hostname)"
echo $(date "+%Y-%m-%d_%H:%M:%S") "Those are the modules loaded"
module list
echo ""

echo $(date "+%Y-%m-%d_%H:%M:%S") "These are the python and mpirun in PATH"
which python
which mpirun
echo ""

echo $(date "+%Y-%m-%d_%H:%M:%S") "Check GPU using nvidia-smi"
nvidia-smi
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo ""


base=$PWD
workdir=$( mktemp -d $TMPDIR/Chenggong_XXXXXXXXXX )

# check from 100 to 1
for part in {100..0}
do
    for win in {0..15}
    do
        if [ ! -f $win/$part/gc.rst7 ]; then
            continue 2
        fi
        if [ ! -f $win/$part/gc.log ]; then
            continue 2
        fi
        if [ ! -f $win/$part/gc.jsonl ]; then
            continue 2
        fi
        # for part >= 1, check RE Step NCYCLE
        if [ $part -ge 1 ] ; then
            if grep -q "RE Step $((NCYCLE-1))" $win/$part/gc.log ; then
                echo "$win/$part/gc.log properly finished, and gc.rst7, gc.log, gc.jsonl exists under $win/$part/"
            else
                continue 2
            fi
        # for part = 0, check RE Step XXX
        else
            if grep -q "RE Step $((0CYCLE-1))" $win/$part/gc.log ; then
                echo "$win/$part/gc.log properly finished, and gc.rst7, gc.log, gc.jsonl exists under $win/$part/"
            else
                echo "$win/$part/gc.log did not finish properly. Exiting."
                exit 1
            fi
        fi
    done
    break
done
part_prev=$part
part=$((part + 1))
echo "Run $part from $part_prev"

echo $(date "+%Y-%m-%d_%H:%M:%S") "Copying files to $workdir"
cp $base/../../system.* $workdir/
cp $script_dir/run_GC_RE.py $workdir/
for win in {0..15}
do
    echo $win
    mkdir -p $workdir/$win/$part $workdir/$win/$part_prev $base/$win/$part/
    cp $win/$part_prev/gc.rst7   $workdir/$win/$part_prev/
    cp $win/$part_prev/gc.jsonl  $workdir/$win/$part_prev/
    cp $win/gc.yml               $workdir/$win/
done
# if "hours reached, stop simulation" can be found in the last 10 lines of 1/$part/gc.log, copy $part as well
if [ -f 1/$part/gc.log ] && tail -n 10 1/$part/gc.log | grep -q "hours reached, stop simulation" ; then
    echo "Detected previous run stopped due to max hours reached. Copying $part data as well."
    echo "In 1/$part/gc.log:"
    tail -n 10 1/$part/gc.log | grep -q "hours reached, stop simulation"
    for win in {0..15}
    do
        for fname in gc.jsonl gc.dcd gc.log gc.rst7
        do
            cp $win/$part/$fname $workdir/$win/$part/
        done
    done
fi
echo $(date "+%Y-%m-%d_%H:%M:%S") "Copying done"

cd $workdir
echo $(date "+%Y-%m-%d_%H:%M:%S") "Running in $workdir"
APPFILE="appfile.$SLURM_JOB_ID"
HOSTFILE="hostfile.$SLURM_JOB_ID"
> "$APPFILE"  # truncate/create
> "$HOSTFILE"

# Split CUDA_VISIBLE_DEVICES on comma into an array
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"

command="./run_GC_RE.py \
    -pdb    system.pdb \
    -system system.xml.gz \
    -multidir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15  \
    -yml gc.yml \
    -maxh MAXHMD \
    -ncycle NCYCLE \
    -start_rst7  $part_prev/gc.rst7 \
    -start_jsonl $part_prev/gc.jsonl \
    -deffnm      $part/gc"

# if there are X GPUs, 
if [[ ${#GPU_ARRAY[@]} -eq 4 ]]; then
    for gpu in "${GPU_ARRAY[@]}"; do
        echo "$HOSTNAME" >> "$HOSTFILE"
        echo "-np 4 -x CUDA_VISIBLE_DEVICES=$gpu $command" >> "$APPFILE"
    done
elif [[ ${#GPU_ARRAY[@]} -eq 2 ]]; then
    for gpu in "${GPU_ARRAY[@]}"; do
        echo "$HOSTNAME" >> "$HOSTFILE"
        echo "-np 8 -x CUDA_VISIBLE_DEVICES=$gpu $command" >> "$APPFILE"
    done
elif [[ ${#GPU_ARRAY[@]} -eq 1 ]]; then
    for gpu in "${GPU_ARRAY[@]}"; do
        echo "$HOSTNAME" >> "$HOSTFILE"
        echo "-np 16 -x CUDA_VISIBLE_DEVICES=$gpu $command" >> "$APPFILE"
    done
else
    echo "Error: Unsupported number of GPUs (${#GPU_ARRAY[@]}). Expected 1, 2, or 4."
    exit 1
fi

echo "$(date "+%Y-%m-%d_%H:%M:%S") $HOSTFILE"
cat "$HOSTFILE"
echo "$(date "+%Y-%m-%d_%H:%M:%S") $APPFILE"
cat "$APPFILE"
echo "#########################################"

mpirun --hostfile $HOSTFILE --app $APPFILE

echo $(date "+%Y-%m-%d_%H:%M:%S") "Simulation finished"

for win in {0..15}
do
    for fname in gc.log gc.dcd gc.jsonl gc.rst7
    do
        cp -v $workdir/$win/$part/$fname $base/$win/$part/
    done
done
echo $(date "+%Y-%m-%d_%H:%M:%S") "Copying done. Cleaning up ..."
cd $base
rm -rf $workdir
echo $(date "+%Y-%m-%d_%H:%M:%S") "All done."
