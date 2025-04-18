import gzip
from pathlib import Path
from typing import Union

from openmm import app, openmm

from .relative import HybridTopologyFactory


def load_sys(sys_file: Union[str, Path]) -> openmm.System:
    """
    Load a serialized OpenMM system from a xml or xml.gz file.

    Parameters
    ----------
    sys_file
        Path to the serialized OpenMM system file (.xml or .xml.gz).

    Returns
    -------
    openmm.System
        The deserialized OpenMM System object.

    Examples
    --------
    >>> from grandfep import utils
    >>> system = utils.load_sys("system.xml.gz")

    """
    # if sys_file is xml.gz
    if str(sys_file).endswith(".gz"):
        with gzip.open(sys_file, 'rt') as f:
            system = openmm.XmlSerializer.deserialize(f.read())
    else:
        with open(sys_file, 'r') as f:
            system = openmm.XmlSerializer.deserialize(f.read())
    return system

def load_top(top_file: Union[str, Path]) -> tuple[app.topology.Topology, Union[app.CharmmPsfFile, app.AmberPrmtopFile]]:
    """
    Load a topology file in PSF (CHARMM) or PRMTOP/PARM7 (AMBER) format.

    After loading a top file, you should call `.setPeriodicBoxVectors()` on the topology object
    to define the box. Without setting the box, trajectory files may lack box information.

    Parameters
    ----------
    top_file
        Path to the topology file (either .psf, .prmtop, or .parm7).

    Returns
    -------
    topology : openmm.app.Topology
        The OpenMM Topology object for use in simulation setup.

    top_object : openmm.app.CharmmPsfFile or openmm.app.AmberPrmtopFile
        The loaded OpenMM file object used to construct the topology.

    Raises
    ------
    ValueError
        If the file format is not supported. Only the extensions .psf, .prmtop, and .parm7 are supported.
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
