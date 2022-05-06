#!/usr/bin/env python3

"""
Metadynamics simulation of Alanine Dipeptide in vacuum with OpenMM and PySAGES.
"""


# %%
import argparse
import os
import sys
import time

import numpy
import pysages

from pysages.collective_variables import DihedralAngle
from pysages.methods import Metadynamics, MetaDLogger
from pysages.utils import try_import

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


# %%
pi = numpy.pi
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
kB = kB.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)

T = 298.15 * unit.kelvin
dt = 2.0 * unit.femtoseconds
adp_pdb = os.path.join(os.pardir, os.pardir, "inputs", "alanine-dipeptide", "adp-vacuum.pdb")


# %%
def generate_simulation(pdb_filename=adp_pdb, T=T, dt=dt):
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField("amber99sb.xml")
    cutoff_distance = 1.0 * unit.nanometer
    topology = pdb.topology

    system = ff.createSystem(
        topology, constraints=app.HBonds, nonbondedMethod=app.PME, nonbondedCutoff=cutoff_distance
    )

    # Set dispersion correction use.
    forces = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        forces[force.__class__.__name__] = force

    forces["NonbondedForce"].setUseDispersionCorrection(True)
    forces["NonbondedForce"].setEwaldErrorTolerance(1.0e-5)

    positions = pdb.getPositions(asNumpy=True)

    integrator = openmm.LangevinIntegrator(T, 1 / unit.picosecond, dt)

    integrator.setRandomNumberSeed(42)

    # platform = openmm.Platform.getPlatformByName(platform)
    # simulation = app.Simulation(topology, system, integrator, platform)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    simulation.reporters.append(app.PDBReporter("output.pdb", 1000))
    simulation.reporters.append(
        app.StateDataReporter("log.dat", 1000, step=True, potentialEnergy=True, temperature=True)
    )

    return simulation


# %%
def get_args(argv):
    available_args = [
        ("well-tempered", "w", bool, 0, "Whether to use well-tempered metadynamics"),
        ("use-grids", "g", bool, 0, "Whether to use grid acceleration"),
        ("log", "l", bool, 0, "Whether to use a callback to log data into a file"),
        ("time-steps", "t", int, 5e5, "Number of simulation steps"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run metadynamics")
    for (name, short, T, val, doc) in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)
    return parser.parse_args(argv)


# %%
def main(argv=[]):
    args = get_args(argv)

    cvs = [DihedralAngle([4, 6, 8, 14]), DihedralAngle([6, 8, 14, 16])]

    height = 1.2  # kJ/mol
    sigma = [0.35, 0.35]  # radians
    deltaT = 5000 if args.well_tempered else None
    stride = 500  # frequency for depositing gaussians
    timesteps = args.time_steps
    ngauss = timesteps // stride  # total number of gaussians

    # Grid for storing bias potential and its gradient
    grid = pysages.Grid(lower=(-pi, -pi), upper=(pi, pi), shape=(50, 50), periodic=True)
    grid = grid if args.use_grids else None

    # Method
    method = Metadynamics(cvs, height, sigma, stride, ngauss, deltaT=deltaT, kB=kB, grid=grid)

    # Logging
    hills_file = "hills.dat"
    callback = MetaDLogger(hills_file, stride) if args.log else None

    tic = time.perf_counter()
    method.run(generate_simulation, timesteps, callback)
    toc = time.perf_counter()
    print(f"Completed the simulation in {toc - tic:0.4f} seconds.")

    return method


# %%
if __name__ == "__main__":
    main(sys.argv[1:])
