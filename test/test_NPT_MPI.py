from pathlib import Path
from sys import stdout

import pytest
from mpi4py import MPI

import numpy as np

from openmm import app, unit, openmm

from grandfep import utils, sampler

platform_ref = openmm.Platform.getPlatformByName('Reference')

nonbonded_Amber = {"nonbondedMethod": app.PME,
                   "nonbondedCutoff": 1.0 * unit.nanometer,
                   "constraints": app.HBonds,
                   }

base = Path(__file__).resolve().parent

@pytest.mark.mpi(minsize=4)
def test_RE():
    print()
    print("# Initiate an NPTSamplerMPI with pure OPC water")
    base = Path(__file__).resolve().parent
    multidir = [base / f"Water_Chemical_Potential/OPC/multidir/{i}" for i in range(1,5)]
    sim_dir = multidir[MPI.COMM_WORLD.Get_rank()]
    mdp = utils.md_params_yml(sim_dir / "md.yml")

    inpcrd = app.AmberInpcrdFile(str(base / "Water_Chemical_Potential/OPC/water_opc.inpcrd"))
    prmtop = app.AmberPrmtopFile(str(base / "Water_Chemical_Potential/OPC/water_opc.prmtop"),
                                 periodicBoxVectors=inpcrd.boxVectors)
    system = prmtop.createSystem(**nonbonded_Amber)

    # customize a system so that 1 water can be controlled by global parameters
    baseGC = sampler.BaseGrandCanonicalMonteCarloSampler(
        system,
        prmtop.topology,
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        2.0 * unit.femtosecond,
        "test_base_Amber.log",
    )

    system = baseGC.system
    system.addForce(openmm.MonteCarloBarostat(mdp.ref_p, mdp.ref_t))

    lambda_dict = {"lambda_gc_vdw"     : mdp.lambda_gc_vdw,
                   "lambda_gc_coulomb" : mdp.lambda_gc_coulomb,
                   }

    npt = sampler.NPTSamplerMPI(
        system=system,
        topology=prmtop.topology,
        temperature=mdp.ref_t,
        collision_rate=1 / mdp.tau_t,
        timestep=mdp.dt,
        log=str(sim_dir / "test_npt.log"),
        # platform=platform_ref,
        rst_file=str(sim_dir / "opc_npt_output.rst7"),
        init_lambda_state = mdp.init_lambda_state,
        lambda_dict = lambda_dict,
    )
    npt.logger.info("MD Parameters:\n" + str(mdp))
    assert mdp.init_lambda_state == npt.rank + 1

    npt.load_rst(str(base / "Water_Chemical_Potential/OPC/eq.rst7"))
    assert npt.size == 4

    reduced_e = npt._calc_full_reduced_energy()
    reduced_e_matrix, re_res = npt.replica_exchange(calc_neighbor_only=False)
    assert reduced_e_matrix.shape == (4, 6)

    # all replicas have been given the same configuration, reduced energy should be the same
    for i in range(0,4):
        print(i)
        assert np.all(np.isclose(reduced_e_matrix[i, :], reduced_e))

    npt.simulation.context.setVelocitiesToTemperature(mdp.ref_t)
    npt.logger.info("MD 25000")
    npt.simulation.step(25000)
    for i in range(10):
        npt.logger.info("MD 200")
        npt.simulation.step(200)
        state_old = npt.simulation.context.getState(getPositions=True, getVelocities=True)
        reduced_e_matrix, re_res = npt.replica_exchange(calc_neighbor_only=True)

        state_new = npt.simulation.context.getState(getPositions=True, getVelocities=True)
        flag_right = (npt.rank, npt.rank+1) in re_res and re_res[(npt.rank, npt.rank+1)][0]
        flag_left  = (npt.rank-1, npt.rank) in re_res and re_res[(npt.rank-1, npt.rank)][0]

        box_old = state_old.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        box_new = state_new.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        pos_old = state_old.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        pos_new = state_new.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        vel_old = state_old.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        vel_new = state_new.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        if flag_right or flag_left:
            # boxVector/positions/velocities should be changed
            assert not np.any(np.isclose( np.diag(box_old), np.diag(box_new) ))
            close_pos = np.sum(np.isclose( pos_old, pos_new ))
            close_vel = np.sum(np.isclose( vel_old, vel_new ))
            assert  close_pos < 50
            assert  close_vel < 10
            npt.logger.info(f"re accept test pass close_pos={close_pos}, close_vel={close_vel}")

        else:
            assert np.all(np.isclose( box_old, box_new))
            assert np.all(np.isclose( pos_old, pos_new ))
            assert np.all(np.isclose( vel_old, vel_new))
            npt.logger.info("re reject test pass")




