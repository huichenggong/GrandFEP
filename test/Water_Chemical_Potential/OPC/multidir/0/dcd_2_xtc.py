#!/usr/bin/env python3

import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter

from MDAnalysis.coordinates.DCD import DCDReader

class dcd_reader(DCDReader):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        self._file.header['nsavc'] = 1

# Load topology only first
u = mda.Universe("../../water_opc.pdb")

# Now add the trajectory with workaround


reader = dcd_reader("opc_npt_output.dcd")
u.trajectory = reader

# Write to XTC
with XTCWriter("opc_npt_output.xtc", n_atoms=u.atoms.n_atoms) as writer:
    for ts in u.trajectory:
        writer.write(u.atoms)






from openmm import app
app.DCDReporter