# Script  
Useful script for all kinds of taskes

## 1. Simulation Scripts
### 1.1. run_NPT_init.py
This script is for NPT density equilibration, no MPI.  

### 1.2. run_NPT_RE.py
This script is for NPT with replica exchange simulation, with MPI.  

### 1.3. run_GC_prep_box.py
This script is for preparing the box for grand canonical simulations. It takes a list of rst7 files and cut a new simulation box from those rst7 files. The extra volume will be converted to ghost water.  

### 1.4. run_GC_RE.py
This script is for grand canonical replica exchange simulation, with MPI.  

## 2. Preparation Scripts
Those scripts are used to prepare/convert files.  

### 2.1. hybrid.py
Generate hybrid topology/openmm.System from state A and state B. State A/B are given in Amber prmtop/rst7 format.  

### 2.2. pair_2_yml.py
Convert pair.dat (pmx AtomMapping output with 1-index) to a mapping.yml (0-index) file.

### 2.3. color_map_pairs.py
Check the atom mapping of 2 ligands. Mapped atoms will be shown in sphere with matched color. pymol index is 1-based-index. This is a pymol script.  
```
pymol -d "run $script_dir/color_map_pairs.py; check_color_mapping lig1.pdb, lig2.pdb, pair.dat"
```
![mapping](../docs/picture/edge_1_to_2.png "Mapping")

### 2.4. fit_and_center.py
Load  pdb and dcd files, fit and center the trajectory. This is a pymol script.  
```
pymol -d "run $script_dir/fit_and_center.py; fit_and_center ../../../system.pdb, gc_??_??.xtc"
```

### 2.5. dcd_2_xtc.py
Convert DCD trajectory files to XTC format.  

### 2.6. remove_ghost.py
Remove ghost atoms from the system, and convert DCD files to XTC file.

## 2. Analysis Scripts
### 2.1. analysis/check_RE.sh
Check the number of exchanges between lambda replicas. Only rank 0 log the exchange information, so only need to check log file for window 0.
```bash
>>> $script_dir/analysis/check_RE.sh rep_?/0/0/gc.log
Log files: rep_0/0/0/gc.log rep_1/0/0/gc.log rep_2/0/0/gc.log
Max replica index in rep_0/0/0/gc.log is 15
   0  x   1  x   2  x   3  x   4  x   5  x   6  x   7  x   8  x   9  x  10  x  11  x  12  x  13  x  14  x  15  
    364    339    343    254    131    209    199    170    203    230    237    253    155    253    261
Minimum/Total exchange: 131/3601
```

### 2.2. analysis/MBAR.py
Estimate free energy difference using MBAR/BAR method.
