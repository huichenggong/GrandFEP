# KcsA_5VKE_SF
```bash
base="/home/chui/E29Project-2023-04-11/131-KcsA/01-model/5VKE_inactivated/03-SF"
cp $base/01-Charmm36m-12Wat/charmm-gui-4500329576/step1_pdbreader.psf ./step1_pdbreader_12WAT.psf 
# cp $base/01-Charmm36m-12Wat/charmm-gui-4500329576/step1_pdbreader.pdb ./step1_pdbreader_12WAT.pdb 

cp $base/02-Charmm36m-8Wat/charmm-gui-4499868385/step1_pdbreader.psf  ./step1_pdbreader_8WAT.psf 
# cp $base/02-Charmm36m-8Wat/charmm-gui-4499868385/step1_pdbreader.pdb  ./step1_pdbreader_8WAT.pdb 

cp $base/03-Charmm36m-12Wat-Cl/charmm-gui-4500545102/step1_pdbreader.psf ./step1_pdbreader_KCL.psf
cp $base/03-Charmm36m-12Wat-Cl/charmm-gui-4500545102/step1_pdbreader.pdb ./step1_pdbreader_KCL.pdb
```

```python
import gzip
from pathlib import Path

import numpy as np
from openmm import openmm, unit, app

def read_params(filename):
    """
    Given a text file which has all the charmm top files line by line
    """
    base_path = Path(filename).parent
    parFiles = []
    for line in open(filename, 'r'):
        if '!' in line:
            line = line.split('!')[0]
        parfile = line.strip()
        if len(parfile) != 0:
            parFiles += [ parfile, ]
    parFiles = [str(base_path/p) for p in parFiles]
    params = app.CharmmParameterSet( *parFiles )
    return params

box_vectors = np.array([[10, 0, 0],
                        [0, 10, 0],
                        [0, 0, 10]]) * unit.nanometer


nboptions = {"nonbondedMethod": app.PME,
             "nonbondedCutoff": 1.2 * unit.nanometer,
             "switchDistance" : 1.0 * unit.nanometer,
             "constraints"    : app.HBonds
            }

params = read_params("/home/chui/E29Project-2023-04-11/131-KcsA/01-model/5VKE_inactivated/01-50mmol/01-Charmm36m/charmm-gui-4457094421/openmm/toppar.str")

psf = app.CharmmPsfFile( "step1_pdbreader_12WAT.psf", periodicBoxVectors=box_vectors)
system = psf.createSystem(params, hydrogenMass=3*unit.amu , **nboptions)

with gzip.open("system_12WAT.xml.gz", 'wt') as f:
    lines = openmm.XmlSerializer.serialize(system)
    f.write(lines)

psf = app.CharmmPsfFile( "step1_pdbreader_8WAT.psf", periodicBoxVectors=box_vectors)
system = psf.createSystem(params, hydrogenMass=3*unit.amu , **nboptions)

with gzip.open("system_8WAT.xml.gz", 'wt') as f:
    lines = openmm.XmlSerializer.serialize(system)
    f.write(lines)

```