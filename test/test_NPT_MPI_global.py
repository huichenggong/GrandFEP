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
    multidir = [base / f"Water_Chemical_Potential/OPC/multidir/{i}" for i in range(2,6)]
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
        dcd_file=str(sim_dir / "opc_npt_output.dcd"),
        init_lambda_state = mdp.init_lambda_state,
        lambda_dict = lambda_dict,
        append=True,
    )
    npt.logger.info("MD Parameters:\n" + str(mdp))
    assert mdp.init_lambda_state == npt.rank + 2

    npt.load_rst(str(base / "Water_Chemical_Potential/OPC/eq.rst7"))
    assert npt.size == 4

    re_res = npt.replica_exchange_global_param(calc_neighbor_only=True)
    npt.logger.info(re_res)
    l_list = npt.comm.allgather(npt.lambda_state_index)
    assert l_list == [3, 2, 5, 4]
    re_res = npt.replica_exchange_global_param(calc_neighbor_only=True)
    npt.logger.info(re_res)
    l_list = npt.comm.allgather(npt.lambda_state_index)
    assert l_list == [4, 2, 5, 3]
    re_res = npt.replica_exchange_global_param(calc_neighbor_only=False)
    l_list = npt.comm.allgather(npt.lambda_state_index)
    assert l_list == [5, 3, 4, 2]

    npt.simulation.context.setVelocitiesToTemperature(mdp.ref_t)
    npt.logger.info("MD 5000")
    npt.simulation.step(5000)

    for i in range(200):
        npt.logger.info("MD 50")
        npt.simulation.step(50)
        l_vdw_old = npt.simulation.context.getParameter("lambda_gc_vdw")
        l_chg_old = npt.simulation.context.getParameter("lambda_gc_coulomb")
        re_decision, exchange = npt.replica_exchange_global_param(calc_neighbor_only=i%3==0)
        l_vdw_new = npt.simulation.context.getParameter("lambda_gc_vdw")
        l_chg_new = npt.simulation.context.getParameter("lambda_gc_coulomb")
        state = npt.simulation.context.getState(getForces=True, enforcePeriodicBox=True)
        force_new = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)

        if exchange:
            assert (l_vdw_old, l_chg_old) != (l_vdw_new, l_chg_new)
            npt.logger.info(f"re accept test pass")

        else:
            assert (l_vdw_old, l_chg_old) == (l_vdw_new, l_chg_new)
            npt.logger.info("re reject test pass")

        if npt.lambda_state_index == 5:
            # The total force on the last 4 atoms should be zero
            assert np.abs(np.sum(force_new[-4:, :])) < 0.3
            assert np.isclose(l_vdw_new, 0.0)
            assert np.isclose(l_chg_new, 0.0)
            npt.logger.info("Dummy atoms force test pass")
        elif npt.lambda_state_index == 4:
            assert np.abs(np.sum(force_new[-4:, :])) > 0.001
            assert np.isclose(l_vdw_new, 0.2)
            assert np.isclose(l_chg_new, 0.0)
            npt.logger.info("Alchem atoms force test pass")
        elif npt.lambda_state_index == 3:
            assert np.abs(np.sum(force_new[-4:, :])) > 0.05
            assert np.isclose(l_vdw_new, 0.4)
            assert np.isclose(l_chg_new, 0.0)
            npt.logger.info("Alchem atoms force test pass")
        else:
            # There should be force on the last 4 atoms
            assert np.abs(np.sum(force_new[-4:, :])) > 0.1
            assert l_vdw_new > 0.4
            assert np.isclose(l_chg_new, 0.0)
            npt.logger.info("Alchem atoms force test pass")

        npt.report_rst()
        npt.report_dcd()







