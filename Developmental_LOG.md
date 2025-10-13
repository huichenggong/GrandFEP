## v0.1.3 REST2  
### Plan
1. REST2
2. AtomMapping
3. Lambda path auto-opt
4. ABFE
5. cli `run_GC_RE`

### REST2
#### 1. charge  
#### 2. vdw  
#### 3. 1-4
#### 4. dihedral
#### 5. bond, angle  
skip

```bash
# create a new branch and switch to it
git checkout -b 0.1.3_dev
git push -u origin 0.1.3_dev
```

Remove the non-vdw atoms from the interaction group reduce to udpate time by order of magnitude. 
Hard code `sigma` `epsilon` into the `CustomNonbondedForce` for the water-water interaction only reduce the time from 
17s to 16s. Split water molecules into several groups saves the time for `updateParametersInContext` 
from 16s to 4s.

### Changes
1. Change the integrator from `openmmtools.integrators.BAOABIntegrator` 
to `openmm.LangevinMiddleIntegrator`, according to the following Reference.  
   1. [GROMACS Stochastic Dynamics and BAOAB Are Equivalent Configurational Sampling Algorithms](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00585)
   2. [Openmm doc LangevinMiddleIntegrator](https://docs.openmm.org/development/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html)
   3. [Unified Efficient Thermostat Scheme for the Canonical Ensemble with Holonomic or Isokinetic Constraints via Molecular Dynamics](https://pubs.acs.org/doi/10.1021/acs.jpca.9b02771)
2. Dihedral with dummy atoms  
Dihedral with dummy atoms are will be selectively scaled. If this
dihedral is an improper or double bonded dihedral, it will not be scaled, 
neither by REST2, nor by `lambda_torsions`. We don't want isomerization of 
a double bond, and we don't want to touch the ring's planarity.
3. XXX


## v0.1.2  
### Problem to be Solved
1. Add trajectory processing tools. We need 1. center the trajectory, 2. remove ghost water, 3. cluster water.
2. Make output writing more efficient.
3. Hybrid system with CMAP.
4. Why does OPC water have force when all the nonbonded interactions are turned off?
5. Lambda path optimization.

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
In RE, if a move is accepted, the global parameter(s) in the MPI rank changes while the coordinate and velocity 
stay in the MPI rank. Only rank 0 collects and writes the trajectory/rst/energy.  

By applying this, RE will only change the global parameter and will neither call `set_ghost_list` nor call `updateParametersInContext`.

### Solution Part 2: split water-water into new CustomNonbondedForce
|        | Old  | Core | New | Fix | Wat | Switch |
|--------|------|------|-----|-----|-----|--------|
| Old    | C1   |      |     |     |     |        |
| Core   | C1   | C1   |     |     |     |        |
| New    | None | C1   | C1  |     |     |        |
| Fix    | C1   | C1   | C1  | C4  |     |        |
| Wat    | C1   | C1   | C1  | C3  | C3  |        |
| Switch | C1   | C1   | C1  | C2  | C2  |  C2    |

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