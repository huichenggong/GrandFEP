# Script  
Useful script for all kinds of taskes

## color_map_pairs.py
Check the atom mapping of 2 ligands. Mapped atoms will be shown in sphere with matched color. pymol index is 1-based-index.  
```
pymol -d "run $script_dir/color_map_pairs.py; check_color_mapping lig1.pdb, lig2.pdb, pair.dat"
```
![mapping](picture/mapping_sample.png "Mapping")