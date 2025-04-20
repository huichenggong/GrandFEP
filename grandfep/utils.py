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

def find_reference_atom_indices(topology : app.Topology, ref_atoms_list: list) -> list:
    """
    Find atom indices in the topology that match the given reference atom definitions.

    Parameters
    ----------
    topology :
        OpenMM topology object

    ref_atoms_list : list of dict
        A list of dictionaries specifying reference atoms.
        Each dictionary can contain any combination of the following keys:

        - ``chain_index`` : int
            Index of the chain in the topology.
        - ``res_name`` : str
            Residue name (e.g., "HOH").
        - ``res_id`` : str
            In openmm topology, res_id is string.
        - ``res_index`` : int
            0-based index of the residue in the topology.
        - ``atom_name`` : str
            Atom name (e.g., "O", "H1").

    Returns
    -------
    list
        A list of integer atom indices matching the provided atom specifications.

    Examples
    --------
    >>> top = "test/KcsA_5VKE_SF/step1_pdbreader_12WAT.psf"
    >>> topology, _ = utils.load_top(top)
    >>> ref_atoms_list = [{"res_id":"71", "res_name":"GLU", "atom_name":"O"}]
    >>> utils.find_reference_atom_indices(topology, ref_atoms_list)
    [21, 164, 307, 450]
    """
    atom_indices = []
    for ref_atom in ref_atoms_list:
        for atom in topology.atoms():
            found_flag = []
            for k, v in ref_atom.items():
                if k == "res_name":
                    found_flag.append(atom.residue.name == v)
                elif k == "res_id":
                    found_flag.append(atom.residue.id == v)
                elif k == "res_index":
                    found_flag.append(atom.residue.index == v)
                elif k == "atom_name":
                    found_flag.append(atom.name == v)
                elif k == "chain_index":
                    found_flag.append(atom.residue.chain.index == v)
                else:
                    raise ValueError("Unknown key: {}".format(k))
            if all(found_flag):
                if atom.index not in atom_indices:
                    atom_indices.append(atom.index)
                else:
                    raise ValueError(f"Duplicate reference atom found: {atom}")
    if len(atom_indices) == 0:
        raise ValueError("No reference atom found.")
    return atom_indices