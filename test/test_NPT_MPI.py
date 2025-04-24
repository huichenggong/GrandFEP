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

@pytest.mark.mpi(minsize=4)
def test_RE():
    print()
    print("# Initiate an NPTSamplerMPI with pure OPC water")
    base = Path(__file__).resolve().parent
    multidir = [base / f"Water_Chemical_Potential/OPC/multidir/{i}" for i in range(1,5)]
    sim_dir = multidir[MPI.COMM_WORLD.Get_rank()]
    mdp = utils.md_params_yml()
    mdp.lambda_gc_vdw = None
    mdp.lambda_gc_coulomb = None
    mdp._read_yml(sim_dir / "md.yml")

    inpcrd = app.AmberInpcrdFile(str(base / "Water_Chemical_Potential/OPC/water_opc.inpcrd"))
    prmtop = app.AmberPrmtopFile(str(base / "Water_Chemical_Potential/OPC/water_opc.prmtop"),
                                 periodicBoxVectors=inpcrd.boxVectors)
    system = prmtop.createSystem(**nonbonded_Amber)
    system.addForce(openmm.MonteCarloBarostat(mdp.ref_p, mdp.ref_t))

    npt = sampler.NPTSamplerMPI(
        system=system,
        topology=prmtop.topology,
        temperature=mdp.ref_t,
        collision_rate=1 / mdp.tau_t,
        timestep=mdp.dt,
        log=str(sim_dir / "test_npt.log"),
        rst_file=str(sim_dir / "opc_npt_output.rst7"),
    )
    npt.logger.info("MD Parameters:\n"+str(mdp))
    assert mdp.init_lambda_state == npt.rank+1

    lambda_dict = {"lambda_gc_vdw"     : mdp.lambda_gc_vdw,
                   "lambda_gc_coulomb" : mdp.lambda_gc_coulomb,
                   }
    npt.set_lambda_dict(mdp.init_lambda_state, lambda_dict)

