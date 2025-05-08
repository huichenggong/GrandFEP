# 0.1.1 Performance update in RE and GC
In the `0.1.0`, `updateParametersInContext` cost ~1 min. This means if a RE/GCMC is accepted, the particular replica will spend 1 min  waiting. There are two possible way to speed this up.  

## Solution Part 1: no configuration exchange in RE
In RE, if a move is accepted, the global parameter(s) in the MPI rank changes while the coordinate and velocity stay in the MPI rank. Each rank has to figure out which file to append during the `report_rst` and `report_dcd`. Rank 0 write the reduced energy to a csv file.  

By applying this, RE will only change the ghobal parameter and will not call `set_ghost_list` nor `updateParametersInContext`.

## Solution Part 2: split water-water into new CustomNonbondedForce
|      | old  | core | new  | fix  | wat  |
|------|------|------|------|------|------|
| old  |  C1  |  X   |  X   |  X   |  X   |
| core |  C1  |  C1  |  X   |  X   |  X   |
| new  | None |  C1  |  C1  |  X   |  X   |
| fix  |  C1  |  C1  |  C1  |  C3  |  X   |
| wat  |  C1  |  C1  |  C1  |  C2  |  C2  |

`C1` uses the same energy expression as before. The `env` group includes `fix` and `wat`, and the `env`-`env` intrection is now included in this `CustomNonbondedForce`.  

`C2` does not need to split state A and B in per particle parameter. It still need `is_real` and `is_switch`, the same as `C1`. It avoids the many global parameters which controls the RBFE and only includes `lambda_gc_vdw` and `lambda_gc_coulomb` for controlling the switching water  

`C3` does not change.  


# 0.1.0 Basic GC FEP functionalities
## 1. Documentation
### 1.1 Install Sphinx and autodoc extention
```bash
pip install sphinx sphinx-autodoc-typehints
mamba install ghp-import # for pushing doc to github
```

### 1.2 Initialize Sphinx in the project
```
sphinx-quickstart docs
```
Edit `docs/conf.py`  
```bash
make html
```

### 1.3 Push the documentation to gh-pages
```
ghp-import -n -p docs/_build/html
```

### 1.4 Check it online
[Documentation](https://huichenggong.github.io/GrandFEP/)

## 2. set up pytest-mpi
```bash
conda install pytest-mpi
```

### 2.1 Run NPT RE test
```bash
mpirun -n 4 python -m pytest --with-mpi test_NPT_MPI.py
```

### 2.2 Run GC RE test
```bash
mpirun -n 4 python -m pytest --with-mpi test/test_GC_MPI.py::test_GC_RE
mpirun -n 8 --use-hwthread-cpus python -m pytest --with-mpi test/test_GC_MPI.py::test_GC_RE
# I only have 4 physical cores here.
```