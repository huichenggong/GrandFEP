# fit_and_center.py
from pathlib import Path

from pymol import cmd
import os

def fit_and_center(pdb_file, dcd_file, align_selection=None, center_selection=None, obj_name=None):
    """
    Load a PDB + DCD, perform intra-frame fit, and center on a selection.

    Parameters
    ----------
    pdb_file : str
        Path to the topology (PDB/PSF + matching atom order for DCD).
    dcd_file : str, list
        Path to the trajectory (DCD).
    align_selection : str
        PyMOL selection used by `intra_fit` (e.g., "name CA", "backbone", "not elem H").
    center_selection : str
        PyMOL selection to center/orient/zoom on after fitting.
    obj_name : str, optional
        Name of the loaded object; defaults to basename of pdb_file without extension.
    """
    if obj_name is None:
        obj_name = os.path.splitext(os.path.basename(pdb_file))[0]

    cmd.reinitialize()  # fresh session
    cmd.set("retain_order", 1)  # important for matching PDB ↔ DCD atom order

    # 1) load
    cmd.load(pdb_file, obj_name)
    base=Path("./")
    for dcd in base.glob(dcd_file):
        print(f"Loading trajectory: {dcd}")
        cmd.load_traj(dcd, obj_name, state=0)

    if align_selection is None:
        # Is there atoms that match "name CA" in the object?
        if cmd.select("ca_sel", f"({obj_name} and name CA)") > 0:
            align_selection = "name CA"
        elif cmd.select("mol_sel", f"({obj_name} and resn MOL)") > 0:
            align_selection = "resn MOL"
        else:
            align_selection = "(not (resn HOH or resn WAT or resn TIP3 or resn TIP4 or resn TIP5 or resn SOL or resn K* or resn NA* or resn CL* ))"
    
    if center_selection is None:
        # Is there atoms that match "resn MOL" in the object?
        if cmd.select("mol_sel", f"({obj_name} and resn MOL)") > 0:
            center_selection = "resn MOL"
        else:
            center_selection = align_selection

    # 2) intra_fit (align each state to state 1 by selection)
    #    If your selection is object-relative, prepend the object for safety.
    sel_fit = f"({obj_name} and ({align_selection}))"
    cmd.intra_fit(sel_fit)

    # 3) center/orient/zoom on a selection (within the same object)
    sel_center = f"({obj_name} and ({center_selection}))"
    cmd.center(sel_center)
    cmd.orient(sel_center)
    cmd.zoom(sel_center)

# Expose as a PyMOL command: fit_and_center pdb, dcd, align_selection, center_selection[, obj_name]
cmd.extend("fit_and_center", fit_and_center)
