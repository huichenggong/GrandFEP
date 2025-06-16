## v0.1.1 Speed up `updateParametersInContext`
### Create a new branch
```bash
# create a new branch and switch to it
git checkout -b 0.1.1_dev

# push the branch to github
git push -u origin 0.1.1_dev

# merge to master
git checkout master
git pull origin master
git merge 0.1.1_dev
git push origin master
```

### Problem to be Solved
In the `0.1.0`, `updateParametersInContext` cost ~1 min. This means if a RE/GCMC is accepted, the particular replica will spend 1 min  waiting. There are two solutions to speed this up.  

### Solution Part 1: no configuration exchange in RE
In RE, if a move is accepted, the global parameter(s) in the MPI rank changes while the coordinate and velocity stay in the MPI rank. Each rank has to figure out which file to append during the `report_rst` and `report_dcd`. Rank 0 write the reduced energy to a csv file.  

By applying this, RE will only change the global parameter and will neither call `set_ghost_list` nor call `updateParametersInContext`.

### Solution Part 2: split water-water into new CustomNonbondedForce
|        | old  | core | new  | fix  | wat  | Switch |
|--------|------|------|------|------|------|--------|
| Old    |  C1  |      |      |      |      |        |
| Core   |  C1  |  C1  |      |      |      |        |
| New    | None |  C1  |  C1  |      |      |        |
| Fix    |  C1  |  C1  |  C1  |  C4  |      |        |
| Wat    |  C1  |  C1  |  C1  |  C3  |  C3  |        |
| Switch |  C1  |  C1  |  C1  |  C2  |  C2  |  C2    |

`C1` uses the same energy expression as before. The `env` group includes `Fix` and `Wat`, and the `env`-`env` intrection is now included in this `CustomNonbondedForce`.  

`C2` does not need to split state A and B in per particle parameter. It still need `is_real` and `is_switch`, the same as `C1`. It avoids the many global parameters which controls the RBFE and only includes `lambda_gc_vdw` and `lambda_gc_coulomb` for controlling the switching water  

`C3` only needs `is_real`, no global parameter

`C4` does not change.  

If water H has no vdw, remove them from the water group also speeds up the update. In this version, for the same HSP90 system, the time for `updateParametersInContext` is ~ 7s.  


## v0.1.0 Basic GC FEP functionalities
### 1. Documentation
#### 1.1 Install Sphinx and autodoc extention
```bash
pip install sphinx sphinx-autodoc-typehints
mamba install ghp-import # for pushing doc to github
pip install --upgrade sphinx_mdinclude # for markdown support
```

#### 1.2 Initialize Sphinx in the project
```
sphinx-quickstart docs
```
Edit `docs/conf.py`  
```bash
make html
```

#### 1.3 Push the documentation to gh-pages
```
ghp-import -n -p docs/_build/html
```

#### 1.4 Check it online
[Documentation](https://huichenggong.github.io/GrandFEP/)

### 2. set up pytest-mpi
```bash
conda install pytest-mpi
```

#### 2.1 Run NPT RE test
```bash
mpirun -n 4 python -m pytest --with-mpi test/test_NPT_MPI.py
```

#### 2.2 Run GC RE test
```bash
mpirun -n 4 python -m pytest --with-mpi test/test_GC_MPI.py::test_GC_RE
mpirun -n 8 --use-hwthread-cpus python -m pytest --with-mpi test/test_GC_MPI.py::test_GC_RE
# I only have 4 physical cores here.
```