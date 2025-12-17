
import os
from pathlib import Path
from pymol import cmd

# load protein and ligands
base=Path(os.environ.get("PWD"))
ff=base.parent.parent.name
# get 4 parents up
for _ in range(4):
    base = base.parent

for edge in base.glob("edge_*_to_*"):
    print(edge)
    cmd.load(edge / ff / "protein/system.pdb", edge.name)

cmd.hide("sphere", "all")
cmd.hide("lines", "all")

with open(base/"center.csv", "r") as f:
    lines = f.readlines()[1:]  # skip header
    for line in lines:
        edge, center = line.strip().split(",")
        for post_fix in [""]:
            edge_name = edge + post_fix
            cmd.show("sphere",   f"/{edge_name}//*/MOL*/{center}")
            cmd.show("sphere",   f"resn HOH and name O within 8.5 of /{edge_name}//*/MOL*/{center}")
            cmd.show("lines",    f"all                 within 8.5 of /{edge_name}//*/MOL*/{center}")
            cmd.show("licorice", f"bymolecule (resn HOH and name O within 12 of /{edge_name}//*/MOL*/{center})")

cmd.set("sphere_scale", 0.5)
cmd.set("cartoon_transparency", 0.5)



