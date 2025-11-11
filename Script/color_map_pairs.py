# color_map_pairs.py
# pymol -d "run $script_dir/color_map_pairs.py; check_color_mapping lig1.pdb, lig2.pdb, pair.dat"
from pymol import cmd
import random
import os


def _read_pairs(path, zero_based=False):    
    pairs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            a, b = line.split()[:2]
            i, j = int(a), int(b)
            if zero_based:
                i += 1  # PyMOL 'index' is 1-based
                j += 1
            pairs.append((i, j))
    return pairs

def check_color_mapping(lig1_pdb, lig2_pdb, pairs_dat):
    """
    Load two ligands, enable grid view, and color mapped atom pairs randomly.

    Parameters
    ----------
    lig1_pdb : str
        Path to first ligand PDB (shown in left grid cell as 'ligA').
    lig2_pdb : str
        Path to second ligand PDB (shown in right grid cell as 'ligB').
    pairs_dat : str
        Path to mapping file with two integer columns (0-based: old new).
    seed : int
        RNG seed for reproducible colors.
    """

    cmd.reinitialize()
    cmd.set('grid_mode', 1)
    cmd.set('retain_order', 1)

    # Load and show sticks
    cmd.load(lig1_pdb, 'ligA')
    cmd.load(lig2_pdb, 'ligB')
    cmd.load(lig1_pdb, 'ligA_elements')
    cmd.load(lig2_pdb, 'ligB_elements')
    cmd.hide('everything', 'ligA')
    cmd.hide('everything', 'ligB')
    cmd.show('sticks', 'ligA or ligB')

    # Read mapping (0-based -> shift to PyMOL index which is 1-based)
    pairs = _read_pairs(pairs_dat, zero_based=False)

    # Color each mapped pair with the same random color
    for k, (idxA, idxB) in enumerate(pairs):
        cname = f'map_{k:03d}'
        rgb = [random.random() for _ in range(3)]
        cmd.set_color(cname, rgb)
        cmd.color(cname,   f'ligA and index {idxA}')
        cmd.color(cname,   f'ligB and index {idxB}')
        cmd.show('sphere', f'ligA and index {idxA}')
        cmd.show('sphere', f'ligB and index {idxB}')

    # Add labels, index
    cmd.label('ligA', 'index')
    cmd.label('ligB', 'index')

    cmd.set('sphere_scale', 0.3)
    cmd.set('label_size',   40)

# expose as a PyMOL command: check_color_mapping lig1, lig2, pairs
cmd.extend('check_color_mapping', check_color_mapping)
