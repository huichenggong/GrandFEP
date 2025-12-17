#!/bin/bash
#
#SBATCH --job-name=JOBNAME
#SBATCH --partition=p10,p12,p16,p20,p24,p32,p08,p40,p06
#SBATCH --get-user-env
#SBATCH --gres=gpu:1              # number of GPUs requested
#SBATCH -t 0:30:00                # hours:min:sec
#SBATCH --dependency=singleton


source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
module add openmpi/gcc/64/4.1.5

echo $(date "+%Y-%m-%d_%H:%M:%S") "This job is running on: $HOSTNAME"
echo ""

echo $(date "+%Y-%m-%d_%H:%M:%S") "Those are the loaded modules:"
module list
echo ""

echo $(date "+%Y-%m-%d_%H:%M:%S") "These are the python and mpirun in PATH"
which python
which mpirun
echo ""



$script_dir/run_GC_prep_box.py \
    -pdb ../../system.pdb \
    -system ../../system.xml.gz \
    -multidir MULTIDIR \
    -yml gc_eq.yml \
    -start_rst7 npt_eq.rst7 \
    -odeffnm DEFFNM \
    -box $(< ../gc_start.dat)
