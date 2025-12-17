#!/bin/bash
#
#SBATCH --job-name=JOBNAME
#SBATCH --partition=p16,p20,p24,p32,p40
#SBATCH --get-user-env
#SBATCH --gres=gpu:NGPU              # number of GPUs requested
#SBATCH -t MAXHMD:30:00                # hours:min:sec
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

echo $(date "+%Y-%m-%d_%H:%M:%S") "Copying files to $workdir"
cp $base/../../system.* $workdir/
cp $script_dir/run_GC_RE.py $workdir/
for win in {0..15}
do
    echo $win
    mkdir -p $workdir/$win/0 $base/$win/0/
    for fname in gc_eq.yml  gc_start.jsonl  gc_start.rst7
    do
        cp $win/$fname $workdir/$win/
    done
done
for win in {0..15}
do
    if [ ! -f $win/0/gc.rst7 ] ; then
        break
    fi
    for fname in gc.rst7 gc.jsonl gc.log gc.dcd
    do
        if [ -f $win/0/$fname ]; then
            cp -v $win/0/$fname $workdir/$win/0/
        fi
    done
done
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
    -yml gc_eq.yml \
    -maxh MAXHMD \
    -ncycle NCYCLE \
    -start_rst7   gc_start.rst7 \
    -start_jsonl  gc_start.jsonl \
    -deffnm       0/gc"

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
        cp -v $win/0/$fname $base/$win/0/
    done
done
echo $(date "+%Y-%m-%d_%H:%M:%S") "Copying done. Cleaning up ..."
cd $base
rm -rf $workdir
echo $(date "+%Y-%m-%d_%H:%M:%S") "All done."
