#!/bin/bash
#
#SBATCH --job-name=JOBNAME
#SBATCH --partition=p16,p20,p24,p32,p40,p08,p12
#SBATCH --get-user-env
#SBATCH --gres=gpu:1              # number of GPUs requested
#SBATCH -t 0:10:00                # hours:min:sec
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
batch_size=4 # 16 windows in batch of 4
for win0 in $(seq 0 $batch_size 15)
do
    win1=$((win0+batch_size-1))
    for win in $(seq $win0 $win1)
    do
        cd $base/$win
        if [ ! -f npt_eq.pdb ]; then
            echo "Run $win"
            $script_dir/run_NPT_init.py -pdb ../../../system.pdb -system ../../../system.xml.gz -nstep 50000 -yml npt_eq.yml -deffnm npt_eq &
            # 50000 * 0.004 ps = 200 ps
        fi
    done
    wait
    date "+%Y-%m-%d_%H:%M:%S"
done
