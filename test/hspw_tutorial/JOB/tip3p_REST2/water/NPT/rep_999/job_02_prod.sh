#!/bin/bash
#
#SBATCH --job-name=JOBNAME
#SBATCH --partition=p16,p20,p24,p32,p40
#SBATCH --get-user-env
#SBATCH --gres=gpu:1              # number of GPUs requested
#SBATCH -t 4:10:00                # hours:min:sec
#SBATCH --dependency=singleton
# EXCLUDE

# use your conda installation path
source /YourCondaInstallationDir/miniforge3/bin/activate grandfep_dev
# set script directory
script_dir="/GrandFEP_Gitdir/Script/"

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
cp $script_dir/run_NPT_RE.py $workdir/
for win in {0..15}
do
    mkdir -p $workdir/$win/0 $base/$win/0/
    cp $win/npt_eq.rst7 $workdir/$win/
    cp $win/npt.yml     $workdir/$win/
    for name in npt.rst7 npt.log npt.dcd
    do
        if [ -f $win/0/$name ]; then
            cp $win/0/$name $workdir/$win/0/ # for appending
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

command="./run_NPT_RE.py \
    -pdb    system.pdb \
    -system system.xml.gz \
    -multidir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15  \
    -yml npt.yml \
    -maxh 4 \
    -ncycle 10000 \
    -start_rst7 npt_eq.rst7 \
    -deffnm 0/npt "

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
    for fname in npt.rst7 npt.log npt.dcd
    do
        cp -v $workdir/$win/0/$fname $base/$win/0/
    done
done
echo $(date "+%Y-%m-%d_%H:%M:%S") "Copying done. Cleaning up ..."
cd $base
rm -rf $workdir
echo $(date "+%Y-%m-%d_%H:%M:%S") "All done."
