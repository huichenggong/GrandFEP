import gzip
import warnings
from pathlib import Path
from typing import Union

import yaml
import numpy as np
import pandas as pd

from openmm import app, openmm, unit
import pymbar

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

def seconds_to_hms(seconds: float) -> tuple[int, int, float]:
    """
    Convert seconds to hours, minutes, and seconds.

    Parameters
    ----------
    seconds
        Time in seconds.

    Returns
    -------
    hours

    minutes

    seconds
    """
    n_hours, remainder   = divmod(seconds, 3600)
    n_minutes, n_seconds = divmod(remainder, 60)
    return int(n_hours), int(n_minutes), n_seconds


def find_mapping(map_list: list, resA: app.topology.Residue, resB: app.topology.Residue) -> tuple[bool, dict]:
    """
    Check if residues A and B match with a mapping in the mapping list.

    Parameter
    ---------
    map_list :
        A list of mapping. For example

        .. code-block:: python

            [{'res_nameA': 'MOL', 'res_nameB': 'MOL',
              'index_map': {1: 1, 0: 2, 2: 0}
    resA :
        residue from state A

    resB :
        residue from state B

    Returns
    -------
    match_flag :
        resA and resB can be matched

    index_map :
        A dictionay mapping the atoms from state A to state B with their 0-index

    """
    for mapping in map_list:
        match_flag = True
        if "res_nameA" in mapping and "res_nameB" in mapping:
            if not (resA.name == mapping["res_nameA"] and resB.name == mapping["res_nameB"]):
                match_flag = False
        if "res_indexA" in mapping and "res_indexB" in mapping:
            if not (resA.index == mapping["res_indexA"] and resB.index == mapping["res_indexB"]):
                match_flag = False
        if match_flag:
            return match_flag, mapping['index_map']
    return False, None


def prepare_atom_map(topologyA: app.Topology, topologyB: app.Topology, map_list: list) -> tuple[dict, dict]:
    """
    With given topology A and B, prepare the atom mapping.

    Parameters
    ----------
    topologyA :
        topology for state A

    topologyB :
        topology for state B

    map_list :
        A list of mapping. For example

        .. code-block:: python

            [{'res_nameA': 'MOL', 'res_nameB': 'MOL',
              'index_map': {1: 1, 0: 2, 2: 0, 25: 26, 22: 29, 23: 28, 24: 27}

    Returns
    -------
    old_to_new_all :
        A dictionay for all the atoms that should map from old (state A) to new (state B)

    old_to_new_core :
        A dictionay for Alchemical (core) atoms that should map from old (state A) to new (state B)
    """
    old_to_new_all = {}  # all the atoms that should map from A to B
    old_to_new_core = {}  # Alchemical atoms that should map from A to B

    res_listA = [res for res in topologyA.residues()]
    res_listB = [res for res in topologyB.residues()]
    for resA, resB in zip(res_listA, res_listB):
        at_listA = [at for at in resA.atoms()]
        at_listB = [at for at in resB.atoms()]

        alchem_map_flag, index_map = find_mapping(map_list, resA, resB)
        if alchem_map_flag:
            for atA_mol_index, atB_mol_index in index_map.items():
                atA_sys_index = at_listA[atA_mol_index].index
                atB_sys_index = at_listB[atB_mol_index].index
                old_to_new_core[atA_sys_index] = atB_sys_index
                old_to_new_all[atA_sys_index] = atB_sys_index
        elif resA.name == resB.name and len(at_listA) == len(at_listB):
            for atA, atB in zip(at_listA, at_listB):
                if not atA.name == atB.name:
                    raise ValueError(f"{atA.name} in {resA.name} cannot map to {atB.name} in {resB.name}")
                old_to_new_all[atA.index] = atB.index
        else:
            raise ValueError(f"{resA} - {resB} Cannot be Mapped")
    return old_to_new_all, old_to_new_core

class md_params_yml:
    """
    Class to manage MD parameters with default values and YAML overrides. This class reads whatever in the yml file, and
    it does not guarantee the correct usage of the parameters anywhere else. The built-in parameters in
    ``self._unit_map`` will be automatically added with units. If you want a unit to be attached, you can add it to
    ``self._unit_map`` before loading the yml file.

    Example:
    --------
    >>> from grandfep import utils
    >>> mdp = utils.md_params_yml("test/Water_Chemical_Potential/OPC/multidir/0/md.yml")

    Attributes:
        integrator (str): Name of the integrator.
        dt (unit.Quantity): Time step. Unit in ps
        maxh (float): The maximum run time. Unit in hour
        nsteps (int): Number of steps.
        nst_dcd (int): Number of steps per dcd trajectory output.
        nst_csv (int): Number of steps per csv energy file output.
        ncycle_dcd (int): Number of cycles per dcd trajectory output.
        ncycle_csv (int): Number of cycles per csv energy file output.
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
        n_propagation (int): Number of propagation steps bewteen each lambda switching.
        init_lambda_state (int): The lambda state index to simulate
        calc_neighbor_only (bool): Whether to calculate the energy of the nearest neighbor only
            when performing replica exchange.

        md_gc_re_protocol (list): MD, Grand Canonical, Replica Exchange protocol. The default is
            ``[("MD", 200),("GC", 1),("MD", 200),("RE", 1),("MD", 200),("RE", 1),("MD", 200),("RE", 1)]``

        system_setting (dict): Nonbonded parameters for the OpenMM createSystem. The default is
            ``{"nonbondedMethod": app.PME, "nonbondedCutoff": 1.0 * unit.nanometer, "constraints": app.HBonds}``.
            openmm has to be imported in the way the the value string can be evaluated.

            >>> from openmm import app, openmm, unit

        sphere_radius (unit.Quantity): The radius of the GCMC sphere. Unit in nm
        lambda_gc_vdw (list): This is the ghobal parameter for controlling the vdw on the switching water. You
            can give a list of lambda values for the path of GC insertion.
        lambda_gc_coulomb (list): This is the ghobal parameter for controlling the Coulomb on the switching water. You
            can give a list of lambda values for the path of GC insertion.
        lambda_angles (list): Lambda
        lambda_bonds (list): Lambda
        lambda_sterics_core (list): Lambda
        lambda_electrostatics_core (list): Lambda
        lambda_sterics_delete (list): Lambda
        lambda_electrostatics_delete (list): Lambda
        lambda_sterics_insert (list): Lambda
        lambda_electrostatics_insert (list): Lambda
        lambda_torsions (list): Lambda

    """

    def __init__(self, yml_file=None):
        # default unit attribute
        self._unit_map = {
            "dt": unit.picoseconds,
            "tau_t": unit.picoseconds,
            "ref_t": unit.kelvin,
            "gen_temp": unit.kelvin,
            "restraint_fc": unit.kilojoule_per_mole / unit.nanometer ** 2,
            "ref_p": unit.bar,
            "surface_tension": unit.bar * unit.nanometer,
            "ex_potential": unit.kilocalorie_per_mole,
            "standard_volume": unit.nanometer ** 3,
            "sphere_radius":unit.nanometer,
        }

        # Default parameter values
        self.integrator = "LangevinIntegrator"
        self.dt = 0.002 * unit.picoseconds
        self.maxh = 1.0
        self.nsteps = 100
        self.nst_dcd = 0
        self.nst_csv = 0
        self.ncycle_dcd = 0
        self.ncycle_csv = 5
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
        self.n_propagation = 20
        self.init_lambda_state = 0
        self.calc_neighbor_only = False
        self.md_gc_re_protocol = [("MD", 200),
                                  ("GC", 1),
                                  ("MD", 200),
                                  ("RE", 1),
                                  ("MD", 200),
                                  ("RE", 1),
                                  ("MD", 200),
                                  ("RE", 1)]
        self.system_setting = {"nonbondedMethod": "app.PME",
                               "nonbondedCutoff": "1.0 * unit.nanometer",
                               "constraints"    : "app.HBonds"}

        self.sphere_radius = 0.0 * unit.nanometer
        self.lambda_gc_vdw = None
        self.lambda_gc_coulomb = None
        self.lambda_angles                = [1.0]
        self.lambda_bonds                 = [1.0]
        self.lambda_electrostatics_core   = [1.0]
        self.lambda_electrostatics_delete = [1.0]
        self.lambda_electrostatics_insert = [1.0]
        self.lambda_sterics_core          = [1.0]
        self.lambda_sterics_delete        = [1.0]
        self.lambda_sterics_insert        = [1.0]
        self.lambda_torsions              = [1.0]


        # Override with YAML file if provided
        if yml_file:
            self._read_yml(yml_file)

    def _read_yml(self, yaml_file):
        """Load parameters from YAML file and override defaults."""
        with open(yaml_file, "r") as file:
            params = yaml.safe_load(file)

        for key, value in params.items():
            setattr(self, key, self._convert_unit(key, value))

    def get_system_setting(self):
        """
        Evaluate system_setting, and return them in a dictionary
        """
        system_setting = {}
        for k, v in self.system_setting.items():
            system_setting[k] = eval(v)
        return system_setting

    def get_lambda_dict(self) -> dict:
        """

        Returns
        -------
        lambda_dict :
            A dictionary of mapping from global parameters to their values in all the sampling states.
        """
        lambda_dict = {}
        for attr in dir(self):
            if attr.startswith("lambda_") and not attr.startswith("lambda_gc_"):
                lambda_dict[attr] = getattr(self, attr)
        return lambda_dict


    def _convert_unit(self, key, value):
        """Handle unit conversion based on parameter key."""

        if key in self._unit_map:
            return value * self._unit_map[key]
        else:
            return value  # For parameters without explicit units

    def __str__(self):
        """Print parameters for easy checking."""
        params = {attr: getattr(self, attr) for attr in dir(self) if (not attr.startswith("_")) and (not attr.startswith("get"))}
        return "\n".join(f"{k}: {v}" for k, v in params.items())

class FreeEAnalysis:
    """
    Class to analyze free energy calculations using BAR/MBAR.

    Parameters
    ----------

    """
    def __init__(self, file_list: list, keyword: str, separator: str, drop_equil: bool=True, begin: int=0):
        self.file_list = file_list
        data_T_all = [self.read_energy(f, keyword=keyword, separator=separator, begin=begin) for f in self.file_list]

        self.temperature = None
        self.kBT = None
        if data_T_all[0][1] is not None:
            # all the temperature should be the same
            t_all = [data[1].value_in_unit(unit.kelvin) for data in data_T_all]
            if not np.all(np.isclose(t_all[1], [t for t in t_all])):
                msg = "\n".join([f"{f}: {t}" for f, t in zip(self.file_list, [data[1] for data in data_T_all])])
                raise ValueError(f"Temperature are not the same in given files:\n"+msg)

            self.temperature = data_T_all[0][1]
            self.kBT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * self.temperature

        self.U_all = [i[0] for i in data_T_all]
        self.u_unco = None
        self.N_k = None
        self.eq_time = None
        self.sub_sample(drop_equil)

    def set_temperature(self, temperature: unit.Quantity):
        """
        Set the temperature for the analysis.

        Parameters
        ----------
        temperature : unit.Quantity
            Temperature in Kelvin, with unit
        """
        self.temperature = temperature
        self.kBT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * self.temperature

    @staticmethod
    def read_energy(log_file: Union[str,Path],
                    keyword: str ="Reduced Energy U_i(x):",
                    separator: str =",",
                    begin: int=0) -> tuple[np.ndarray, unit.Quantity]:
        """
        Parse the energy from a file

        Parameters
        ----------
        log_file
            File to be parsed

        keyword:
           The keyword to locate the line. The energy should immediately follow this keyword.

        separator:
            The separator to use between energy in the file

        begin:
            The first frame to be analyzed. The first frame is 0.

        Returns
        -------
        e_array : np.array
            The energy array.

        temperature : unit.Quantity
            The temperature in Kelvin. If not found, None is returned.

        """
        with open(log_file) as f:
            lines = f.readlines()
        e_array = []
        for i, l in enumerate(lines):
            if keyword in l:
                e_string = l.rstrip().split(keyword)[-1]
                e_array_tmp = np.array([float(energy) for energy in e_string.split(separator)]) # reduced energy in kBT
                e_array.append(e_array_tmp)
        e_array = np.array(e_array[begin:])

        temperature = None
        for l in lines:
            if "T   =" in l:
                temperature = float(l.split()[-2]) * unit.kelvin

        return e_array, temperature

    def sub_sample(self, drop_equil: bool = True):

        n_sample, n_ham = self.U_all[0].shape
        N_k = np.zeros(n_ham, dtype=np.int64)
        eq_time = np.zeros(n_ham, dtype=np.int64)
        u_unco = []
        for i, U_series in enumerate(self.U_all):
            n_equil, g, neff_max = pymbar.timeseries.detect_equilibration(U_series[:, i])
            if not drop_equil:
                n_equil = 0
            U_equil = U_series[n_equil:, :]
            indices = pymbar.timeseries.subsample_correlated_data(U_equil[:, i], g=g)
            u_unco.append(U_equil[indices, :])
            N_k[i] = len(indices)
            eq_time[i] = n_equil
        self.u_unco, self.N_k, self.eq_time = u_unco, N_k, eq_time
        return u_unco, N_k, eq_time

    def print_uncorrelate(self):
        """
        Print the number of uncorrelated sample in each file
        """
        max_width = max(len(str(fname)) for fname in self.file_list)
        max_width = max(max_width, 9)
        print(f"{'File Name':<{max_width}} |  N/N_all   | Equil")
        print("-" * (max_width + 20))  # Add a separator line
        N = [len(u) for u in self.U_all]
        for fname, n, n_all, eq0 in zip(self.file_list, self.N_k, N, self.eq_time):
            print(f"{str(fname):<{max_width}} | {n:>4d}/{n_all:<4d} | {eq0:4d}")

    def mbar_U_all(self):
        kBT_val = self.kBT.value_in_unit(unit.kilocalorie_per_mole)
        u_unco = self.u_unco
        N_k = self.N_k
        mbar = pymbar.MBAR(np.vstack(u_unco).T, N_k)
        res = mbar.compute_free_energy_differences()
        dG = res["Delta_f"] * kBT_val
        dG_err = res["dDelta_f"] * kBT_val

        res_format = []
        for i in range(len(dG) - 1):
            res_format.append([dG[i, i + 1], dG_err[i, i + 1]])

        overlap = mbar.compute_overlap()["matrix"]
        for i in range(len(dG) - 1):
            res_format[i].append(overlap[i, i+1])
        return dG, dG_err, np.array(res_format)

    def bar_U_all(self):
        kBT_val = self.kBT.value_in_unit(unit.kilocalorie_per_mole)
        u_unco = self.u_unco
        dG = np.zeros((len(u_unco), len(u_unco)))
        dG_err = np.zeros((len(u_unco), len(u_unco)))
        res_format = []
        n_states = len(u_unco)
        for i in range(n_states - 1):
            u_F = u_unco[i][:, i + 1] - u_unco[i][:, i]
            u_R = u_unco[i + 1][:, i] - u_unco[i + 1][:, i + 1]
            res_tmp = pymbar.other_estimators.bar(u_F, u_R)
            over_lab = 0
            try:
                over_lab = pymbar.other_estimators.bar_overlap(u_F, u_R)
            except:
                warnings.warn("pymbar fail to compute BAR overlap")

            res_format.append([res_tmp['Delta_f'] * kBT_val, res_tmp['dDelta_f'] * kBT_val, over_lab])
            dG[i, i + 1] = res_tmp['Delta_f'] * kBT_val
            dG_err[i, i + 1] = res_tmp['dDelta_f'] * kBT_val
        # fill in the rest of dG and dG_err
        for i in range(n_states):
            for j in range(i + 2, n_states):
                dG[i, j] = dG[i, j - 1] + dG[j - 1, j]
                dG_err[i, j] = np.sqrt(dG_err[i, j - 1] ** 2 + dG_err[j - 1, j] ** 2)

        # Fill lower triangular part
        for i in range(n_states):
            for j in range(i):
                dG[i, j] = -dG[j, i]
                dG_err[i, j] = dG_err[j, i]
        return dG, dG_err, np.array(res_format)

    @staticmethod
    def print_res_all(res_all):
        print(f" A - B :   ", end="")
        for k, v in res_all.items():
            print(f"{k:23}", end="")
        print()

        print("-" * (10 + len(res_all) * 23))

        for i in range(len(v[-1])):
            print(f"{i:2d} -{i + 1:2d} :", end="")
            for k, (dG, dG_err, v) in res_all.items():
                print(f" {v[i, 0]:7.3f} +- {v[i, 1]:6.3f} {v[i, 2]:4.2f}", end="")
            print()

        print("-" * (10 + len(res_all) * 23))
        print("Total  :", end="")
        for k, (dG, dG_err, v) in res_all.items():
            print(f" {dG[0, -1]:7.3f} +- {dG_err[0, -1]:6.3f}     ", end="")
        print()
