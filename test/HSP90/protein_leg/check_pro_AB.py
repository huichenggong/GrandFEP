from pymol import cmd

cmd.load("2xab/10_complex_tleap.pdb", "state_A")
cmd.load("2xjg/10_complex_tleap.pdb", "state_B")
cmd.hide("spheres",)
cmd.hide("cartoon",)

cmd.center("resn MOL")
cmd.orient("resn MOL")

cmd.set("retain_order", 1)
cmd.label("resn MOL", "index") # pymol uses 1-based index
cmd.set("label_size", 50)
# grid mode
cmd.set("grid_mode", 1)
