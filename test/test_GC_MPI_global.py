from pathlib import Path
from sys import stdout

import pytest
from mpi4py import MPI

import numpy as np

from openmm import app, unit, openmm

from grandfep import utils, sampler

def load_amber_sys(inpcrd_file, prmtop_file, nonbonded_settings):
    """
    Load Amber system from inpcrd and prmtop file.

    Parameters
    ----------
    inpcrd_file : str

    :param prmtop_file:

    :return: (inpcrd, prmtop, sys)
    Returns
    -------
        inpcrd: openmm.app.AmberInpcrdFile
        prmtop: openmm.app.AmberPrmtopFile
        sys: openmm.System
    """
    inpcrd = app.AmberInpcrdFile(str(inpcrd_file))
    prmtop = app.AmberPrmtopFile(str(prmtop_file),
                                 periodicBoxVectors=inpcrd.boxVectors)
    sys = prmtop.createSystem(**nonbonded_settings)
    return inpcrd, prmtop, sys

platform_ref = openmm.Platform.getPlatformByName('Reference')

nonbonded_Amber = {"nonbondedMethod": app.PME,
                   "nonbondedCutoff": 1.0 * unit.nanometer,
                   "constraints": app.HBonds,
                   }

base = Path(__file__).resolve().parent

@pytest.mark.mpi(minsize=4)
def test_GC_RE():
    print()
    print("# Initiate an NoneqGrandCanonicalMonteCarloSamplerMPI with CH4 to C2H6")
    base = Path(__file__).resolve().parent
    multidir = [base / f"CH4_C2H6/multidir/{i}" for i in range(9)]
    sim_dir = multidir[MPI.COMM_WORLD.Get_rank()]
    mdp = utils.md_params_yml(sim_dir / "md.yml")

    nonbonded_settings = mdp.get_system_setting()

    inp = app.AmberInpcrdFile(base / "CH4_C2H6/multidir/eq.rst7")
    inpcrd0, prmtop0, sys0 = load_amber_sys(
        base / f"CH4_C2H6/lig0/05_solv.inpcrd",
        base / f"CH4_C2H6/lig0/05_solv.prmtop", nonbonded_settings)
    inpcrd1, prmtop1, sys1 = load_amber_sys(
        base / f"CH4_C2H6/lig1/05_solv.inpcrd",
        base / f"CH4_C2H6/lig1/05_solv.prmtop", nonbonded_settings)
    old_to_new_atom_map = {0: 0}
    for i in range(5, 1394):
        old_to_new_atom_map[i] = i + 3

    old_to_new_core_atom_map = {0: 0, }

    h_factory = utils.HybridTopologyFactory(
        sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
        old_to_new_atom_map,  # All atoms that should map from A to B
        old_to_new_core_atom_map,  # Alchemical Atoms that should map from A to B
        use_dispersion_correction=True,
        softcore_LJ_v2=False)

    system = h_factory.hybrid_system
    topology = h_factory.omm_hybrid_topology
    positions = inp.positions

    topology.setPeriodicBoxVectors(inp.boxVectors)
    system.setDefaultPeriodicBoxVectors(*inp.boxVectors)

    lambda_dict = mdp.get_lambda_dict()
    ngcmc = sampler.NoneqGrandCanonicalMonteCarloSamplerMPI(
        system = system,
        topology = topology,
        temperature = mdp.ref_t,
        collision_rate = 1 / mdp.tau_t,
        timestep = mdp.dt,
        log = str(sim_dir / "md.log"),
        # platform = platform_ref,
        water_resname = "HOH",
        water_O_name = "O",
        position = positions,
        chemical_potential = mdp.ex_potential,
        standard_volume = mdp.standard_volume,
        sphere_radius = mdp.sphere_radius,
        reference_atoms = [0],
        rst_file = str(sim_dir / "md.rst7"),
        dcd_file = str(sim_dir / "md.dcd"),
        append_dcd = False,
        jsonl_file = str(sim_dir / "md.jsonl"),
        init_lambda_state = mdp.init_lambda_state,
        lambda_dict = lambda_dict
    )


    ngcmc.set_ghost_list([10, 11, 12])

    ngcmc.logger.info("MD Parameters:\n" + str(mdp))
    assert mdp.init_lambda_state == ngcmc.rank
    assert ngcmc.lambda_state_index == ngcmc.rank

    ngcmc.load_rst(base / "CH4_C2H6/multidir/eq.rst7")
    reduced_e = ngcmc._calc_full_reduced_energy()
    reduced_e_matrix, re_res = ngcmc.replica_exchange(calc_neighbor_only=False)
    assert reduced_e_matrix.shape == (ngcmc.size, 9)

    # all replicas have been given the same configuration, reduced energy should be the same
    for i in range(0, ngcmc.size):
        print(i)
        assert np.all(np.isclose(reduced_e_matrix[i, :], reduced_e))
    for acc, ratio in re_res.values():
        assert acc
        assert ratio > 0.99

    ngcmc.set_ghost_list([10+i for i in range(ngcmc.rank)]) # set different ghost_list

    re_res = ngcmc.replica_exchange_global_param(calc_neighbor_only=True)
    l_list = ngcmc.comm.allgather(ngcmc.lambda_state_index)
    assert l_list == [0, 2, 1, 3]
    re_res = ngcmc.replica_exchange_global_param(calc_neighbor_only=True)
    l_list = ngcmc.comm.allgather(ngcmc.lambda_state_index)
    assert l_list == [1, 3, 0, 2]
    re_res = ngcmc.replica_exchange_global_param(calc_neighbor_only=False)
    l_list = ngcmc.comm.allgather(ngcmc.lambda_state_index)
    assert l_list == [2, 3, 0, 1]

    ngcmc.load_rst(str(sim_dir / "md.rst7"))
    ngcmc.simulation.minimizeEnergy()
    ngcmc.simulation.context.setVelocitiesToTemperature(mdp.ref_t)
    ngcmc.logger.info("MD 1000")
    ngcmc.simulation.step(1000)

    for i in range(30):
        ngcmc.logger.info("MD 50")
        ngcmc.simulation.step(50)
        l_angle_old = ngcmc.simulation.context.getParameter("lambda_angles")
        l_bonds_old = ngcmc.simulation.context.getParameter("lambda_bonds")
        re_decision, exchange = ngcmc.replica_exchange_global_param(calc_neighbor_only=i % 3 == 0)
        l_angle_new = ngcmc.simulation.context.getParameter("lambda_angles")
        l_bonds_new = ngcmc.simulation.context.getParameter("lambda_bonds")
        state = ngcmc.simulation.context.getState(getForces=True, enforcePeriodicBox=True)
        # force_new = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

        if exchange:
            assert (l_angle_old, l_bonds_old) != (l_angle_new, l_bonds_new)
            ngcmc.logger.info(f"re accept test pass")

        else:
            assert (l_angle_old, l_bonds_old) == (l_angle_new, l_bonds_new)
            ngcmc.logger.info("re reject test pass")





