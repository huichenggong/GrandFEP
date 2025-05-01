#!/usr/bin/env python

from pathlib import Path
import argparse
import sys
import time

import numpy as np
from mpi4py import MPI

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

mdp = utils.md_params_yml("0/md.yml")

nonbonded_settings = mdp.get_system_setting()

inpcrd0, prmtop0, sys0 = load_amber_sys(
    "../lig0/05_solv.inpcrd",
    "../lig0/05_solv.prmtop", nonbonded_settings)
inpcrd1, prmtop1, sys1 = load_amber_sys(
    "../lig1/05_solv.inpcrd",
    "../lig1/05_solv.prmtop", nonbonded_settings)
old_to_new_atom_map = {0: 0}
for i in range(5, 1394):
    old_to_new_atom_map[i] = i + 3

old_to_new_core_atom_map = {0: 0,}

h_factory = utils.HybridTopologyFactory(
    sys0, inpcrd0.getPositions(), prmtop0.topology, sys1, inpcrd1.getPositions(), prmtop1.topology,
    old_to_new_atom_map,      # All atoms that should map from A to B
    old_to_new_core_atom_map, # Alchemical Atoms that should map from A to B
    use_dispersion_correction=True,
    softcore_LJ_v2=False)

system = h_factory.hybrid_system
topology = h_factory.omm_hybrid_topology
positions = h_factory.hybrid_positions

topology.setPeriodicBoxVectors(inpcrd0.boxVectors)

system.addForce(openmm.MonteCarloBarostat(mdp.ref_p*15, mdp.ref_t, mdp.nstpcouple))

npt = sampler.NPTSampler(
    system=system,
    topology=topology,
    temperature=mdp.ref_t,
    collision_rate=1 / (1.0 * unit.picoseconds),
    timestep=2 * unit.femtoseconds,
    log="eq.log",
    rst_file="eq.rst7",
)

state_reporter_std = app.StateDataReporter( sys.stdout, 2000, step=True, temperature=True, density=True, speed=True)
npt.simulation.reporters.append(state_reporter_std)

npt.simulation.context.setPeriodicBoxVectors(*inpcrd0.boxVectors)
npt.simulation.context.setPositions(positions)
npt.logger.info(f"Minimize energy")
npt.simulation.minimizeEnergy()

npt.logger.info(f"Set random velocities to {mdp.gen_temp}")
npt.simulation.context.setVelocitiesToTemperature(mdp.gen_temp)

npt.logger.info("MD 20000")
npt.simulation.step(20000)
npt.logger.info("Miniization again")
npt.simulation.minimizeEnergy()
npt.report_rst()
