# GrandFEP
## 1. Installation
### 1.1 Prepare Env
```bash
mamba env create -f env.yml # edit cuda and MPI according to cluster
# or
mamba create -n grandfep 
```

### 1.2 Later on the cluster
```bash
source /home/NAME/SOFTWARE/miniforge3/bin/activate grandfep
module add openmpi4/gcc/4.1.5 # only as an example
which mpirun                  # check if the correct mpirun is used
```

### 1.3 Jupyter kernel
```bash
python -m ipykernel install --user --name grandfep # add
jupyter kernelspec uninstall grandfep              # remove
```