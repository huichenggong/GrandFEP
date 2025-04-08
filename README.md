# GrandFEP
## 1. Installation
### 1.1 Prepare Env
```bash
mamba env create -f env.yml # edit cuda and MPI according to cluster
# or
mamba create -n grandfep_env python=3.12 numpy scipy pandas openmm openmmtools pymbar-core openmpi=4.1.5 mpi4py parmed cudatoolkit=11.8
# also remember to check the cuda with your driver, and MPI on your cluster
```

### 1.2 Later on the cluster
```bash
source /home/NAME/SOFTWARE/miniforge3/bin/activate grandfep
module add openmpi4/gcc/4.1.5 # only as an example
which mpirun                  # check if the correct mpirun is used
```

### 1.3 Jupyter kernel
```bash
python -m ipykernel install --user --name grandfep_env # add
jupyter kernelspec uninstall grandfep_env              # remove
```