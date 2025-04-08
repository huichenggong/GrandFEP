import gzip
from pathlib import Path

from openmm import app, openmm

from .relative import HybridTopologyFactory


def load_sys(sys_file):
    """
    Load serialized system file

    :param sys_file: (str or pathlib.Path)

        X.xml.gz or X.xml. Serialized system file.

    :return: openmm.System

    """
    # if sys_file is xml.gz
    if str(sys_file).endswith(".gz"):
        with gzip.open(sys_file, 'rt') as f:
            system = openmm.XmlSerializer.deserialize(f.read())
    else:
        with open(sys_file, 'r') as f:
            system = openmm.XmlSerializer.deserialize(f.read())
    return system

def load_top(top_file):
    """
    Load topology file. It can be psf(Charmm) or parm7/prmtop(Amber). remember to run topology.setPeriodicBoxVectors.
    if you provide a charmm psf file. If you don't set the box for topology, later trajectory will not have box information.

    :param top_file: (str or pathlib.Path)

        psf or parm7/prmtop file

    :return: (topology, psf/prmtop)

        openmm.Topology: The OpenMM topology object.

        Union[openmm.app.PSFFile, openmm.app.AmberPrmtopFile]: The original OpenMM topology file object.

    """
    top_file = str(top_file)
    if top_file.endswith(".psf"):
        psf = app.CharmmPsfFile(top_file)
        return psf.topology, psf
    elif top_file.endswith(".parm7") or top_file.endswith(".prmtop"):
        prmtop = app.AmberPrmtopFile(top_file)
        return prmtop.topology, prmtop
    else:
        raise ValueError(f"Topology file {top_file} is not supported. Only psf, parm7, and prmtop are supported.")
