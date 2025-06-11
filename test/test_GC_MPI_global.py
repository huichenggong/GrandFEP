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

    ngcmc.load_rst(str(sim_dir / "md.rst7"))
    ngcmc.simulation.minimizeEnergy()
    ngcmc.simulation.context.setVelocitiesToTemperature(mdp.ref_t)
    ngcmc.logger.info("MD 1000")
    ngcmc.simulation.step(1000)

    for calc_neighbor_only in [True, False]:
        for i in range(10):
            ngcmc.logger.info("MD 200")
            ngcmc.simulation.step(200)

            state_old = ngcmc.simulation.context.getState(getPositions=True, getVelocities=True)
            glist_old = ngcmc.get_ghost_list()
            pos_old = state_old.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            vel_old = state_old.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)

            reduced_e_matrix, re_res = ngcmc.replica_exchange(calc_neighbor_only=calc_neighbor_only)
            ngcmc.check_ghost_list()

            state_new = ngcmc.simulation.context.getState(getPositions=True, getVelocities=True)
            glist_new = ngcmc.get_ghost_list()
            pos_new = state_new.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            vel_new = state_new.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)

            flag_right = (ngcmc.rank, ngcmc.rank + 1) in re_res and re_res[(ngcmc.rank, ngcmc.rank + 1)][0]
            flag_left = (ngcmc.rank - 1, ngcmc.rank) in re_res and re_res[(ngcmc.rank - 1, ngcmc.rank)][0]
            if flag_right or flag_left:
                # ghost_list/positions/velocities should be changed
                assert glist_old != glist_new
                close_pos = np.sum(np.isclose( pos_old, pos_new ))
                close_vel = np.sum(np.isclose( vel_old, vel_new ))
                assert  close_pos < 50
                assert  close_vel < 10
                ngcmc.logger.info(f"re accept test pass close_pos={close_pos}, close_vel={close_vel}")

            else:
                assert glist_old == glist_new
                assert np.all(np.isclose( pos_old, pos_new ))
                assert np.all(np.isclose( vel_old, vel_new))
                ngcmc.logger.info("re reject test pass")



