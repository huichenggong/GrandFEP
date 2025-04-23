import gzip
from pathlib import Path
from typing import Union

import yaml
import numpy as np

from openmm import app, openmm, unit

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
    .. code-block:: python
        :linenos:

        from grandfep import utils
        system = utils.load_sys("system.xml.gz")

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

    ref_atoms_list : list
        A list of dictionaries specifying reference atoms.
        Each dictionary can contain any combination of the following keys:

        - ``chain_index``: int
            Index of the chain in the topology.
        - ``res_name``: str
            Residue name (e.g., "HOH").
        - ``res_id``: str
            In openmm topology, res_id is string.
        - ``res_index``: int
            0-based index of the residue in the topology.
        - ``atom_name``: str
            Atom name (e.g., "O", "H1").

    Returns
    -------
    list
        A list of integer atom indices matching the provided atom specifications.

    Examples
    --------
    >>> from grandfep import utils
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

def random_rotation_matrix() -> np.ndarray:
    """
    Generate a random rotation matrix using Shoemake method.

    Returns
    -------
    np.ndarray :
        A 3x3 rotation matrix.

    Examples
    ---------
    .. code-block:: python
        :linenos:

        import numpy as np
        import matplotlib.pyplot as plt
        from grandfep import utils
        def gen_random_vec():
            axis = np.random.normal(0, 1, 3)
            axis /= np.linalg.norm(axis)
            return axis
        res_new = []
        res_ref = []
        vec_init = gen_random_vec() # regardless of the initial vector, the rotated vector should have uniform distribution on x,y,z
        for i in range(100000):
            rot_matrix = utils.random_rotation_matrix()
            res_new.append(np.dot(rot_matrix, vec_init))
            res_ref.append(gen_random_vec())
        res_new = np.array(res_new)
        res_ref = np.array(res_ref)
        fig, axes = plt.subplots(2, 3, dpi=300, figsize = (9,6))
        for res, ax_list in zip([res_new, res_ref], axes):
            for i, ax in enumerate(ax_list):
                ax.hist(res[:, i], orientation='horizontal', density=True)

    """
    u1, u2, u3 = np.random.rand(3)
    x = np.sqrt(1-u1) * np.sin(2*np.pi*u2)
    y = np.sqrt(1-u1) * np.cos(2*np.pi*u2)
    z = np.sqrt(u1) * np.sin(2*np.pi*u3)
    w = np.sqrt(u1) * np.cos(2*np.pi*u3)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def random_rotation_matrix_protoms():
    """
    Copied from https://github.com/essex-lab/grand/blob/master/grand/utils.py. Possibly be wrong.

    Returns
    -------
    np.ndarray :
        A 3x3 rotation matrix.
    """
    # First generate a random axis about which the rotation will occur
    rand1 = rand2 = 2.0

    while (rand1**2 + rand2**2) >= 1.0:
        rand1 = np.random.rand()
        rand2 = np.random.rand()
    rand_h = 2 * np.sqrt(1.0 - (rand1**2 + rand2**2))
    axis = np.array([rand1 * rand_h, rand2 * rand_h, 1 - 2*(rand1**2 + rand2**2)])
    axis /= np.linalg.norm(axis)

    # Get a random angle
    theta = np.pi * (2*np.random.rand() - 1.0)

    # Simplify products & generate matrix
    x, y, z = axis[0], axis[1], axis[2]
    x2, y2, z2 = axis[0]*axis[0], axis[1]*axis[1], axis[2]*axis[2]
    xy, xz, yz = axis[0]*axis[1], axis[0]*axis[2], axis[1]*axis[2]
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[cos_theta + x2*(1-cos_theta),   xy*(1-cos_theta) - z*sin_theta, xz*(1-cos_theta) + y*sin_theta],
                           [xy*(1-cos_theta) + z*sin_theta, cos_theta + y2*(1-cos_theta),   yz*(1-cos_theta) - x*sin_theta],
                           [xz*(1-cos_theta) - y*sin_theta, yz*(1-cos_theta) + x*sin_theta, cos_theta + z2*(1-cos_theta)  ]])

    return rot_matrix

class MDParams:
    """
    Class to manage MD parameters with default values and YAML overrides.

    Attributes:
        integrator (str): Name of the integrator.
        dt (unit.Quantity): Time step. Unit in ps
        maxh (float): The maximum run time. Unit in hour
        nsteps (int): Number of step.
        nstdcd (int): Number of step per dcd trajectory output.
        nstenergy (int): Number of step per csv energy file output.
        tau_t (unit.Quantity): Temperature coupling time constant. Unit in ps
        ref_t (unit.Quantity): Reference temperature. Unit in K
        gen_vel (bool): Generate velocities.
        gen_temp (unit.Quantity): Temperature for random velocity generation. Unit in K
        restraint (bool): Whether to apply restraints.
        restraint_fc (unit.Quantity): Restraint force constant. Unit in kJ/mol/nm^2
        pcoupltype (str): Specifies the kind of pressure coupling used. MonteCarloBarostat, MonteCarloMembraneBarostat
        ref_p (unit.Quantity): Reference pressure. Unit in bar
        nstpcouple (int): Pressure coupling frequency.
        surface_tension (unit.Quantity): Surface tension. Unit in bar*nm
        ex_potential (unit.Quantity): Excess potential in GC. Unit in kcal/mol
        standard_volume (unit.Quantity): Standard volume in GC. Unit in nm^3
        calc_neighbor_only (bool): Whether to calculate the energy of the nearest neighbor only
            when performing replica exchange.
        md_gc_re_protocol (list): MD, Grand Canonical, Replica Exchange protocol. The default is
            ``[("MD", 200),("GC", 1),("MD", 200),("RE", 1),("MD", 200),("RE", 1),("MD", 200),("RE", 1)]``

    """

    def __init__(self, yml_file=None):
        # Default parameter values
        self.integrator = "LangevinIntegrator"
        self.dt = 0.002 * unit.picoseconds
        self.maxh = 1.0
        self.nsteps = 100
        self.nstdcd = 5000
        self.nstenergy = 1000
        self.tau_t = 1.0 * unit.picoseconds
        self.ref_t = 300.0 * unit.kelvin
        self.gen_vel = True
        self.gen_temp = 300.0 * unit.kelvin
        self.restraint = False
        self.restraint_fc = 1000.0  * unit.kilojoule_per_mole / unit.nanometer**2
        self.pcoupltype = None
        self.ref_p = 1.0 * unit.bar
        self.nstpcouple = 25
        self.surface_tension = 0.0 * unit.bar * unit.nanometer
        self.ex_potential = -6.314 * unit.kilocalorie_per_mole # +- 0.022
        self.standard_volume = 2.96299369e-02 * unit.nanometer**3
        self.calc_neighbor_only = True
        self.md_gc_re_protocol = [("MD", 200),
                                  ("GC", 1),
                                  ("MD", 200),
                                  ("RE", 1),
                                  ("MD", 200),
                                  ("RE", 1),
                                  ("MD", 200),
                                  ("RE", 1)]

        # Override with YAML file if provided
        if yml_file:
            self._read_yml(yml_file)

    def _read_yml(self, yaml_file):
        """Load parameters from YAML file and override defaults."""
        with open(yaml_file, "r") as file:
            params = yaml.safe_load(file)

        # Override default attributes only if provided in YAML
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, self._convert_unit(key, value))
            else:
                raise ValueError(f"Parameter '{key}' is not valid.")

    def _convert_unit(self, key, value):
        """Handle unit conversion based on parameter key."""
        unit_map = {
            "dt": unit.picoseconds,
            "tau_t": unit.picoseconds,
            "ref_t": unit.kelvin,
            "gen_temp": unit.kelvin,
            "restraint_fc": unit.kilojoule_per_mole / unit.nanometer ** 2,
            "ref_p": unit.bar,
            "surface_tension": unit.bar * unit.nanometer,
            "ex_potential": unit.kilocalorie_per_mole,
            "standard_volume": unit.nanometer ** 3,
        }

        if key in unit_map:
            return value * unit_map[key]
        else:
            return value  # For parameters without explicit units

    def __str__(self):
        """Print parameters for easy checking."""
        params = {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith("_")}
        return "\n".join(f"{k}: {v}" for k, v in params.items())
