#!/usr/bin/env python3


from pysages.approxfun import compute_mesh
from pysages.backends import SamplingContext
from pysages.colvars import Projection_on_Axis_mobile

# %%
from pysages.methods import (
    CVRestraints,
    Funnel_MetadLogger,
    Funnel_Metadynamics,
    get_funnel_force,
)
from pysages.utils import try_import

# %%
# from IPython import get_ipython
openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")
import math

import dill as pickle
import jax.numpy as np

# import numpy as np
import matplotlib.pyplot as plt
import numpy
from openmm import *
from openmm.unit import *

# ParmEd Imports
from parmed import load_file
from parmed import unit as u
from parmed.openmm import NetCDFReporter, StateDataReporter

import pysages

# %%
pi = numpy.pi
kB = 0.008314462618


# %%
def generate_simulation(T=298.15 * kelvin, dt=1.0 * femtoseconds):
    print("Loading AMBER files...")
    ala2_solv = load_file("complex-wat.prmtop", "complex-wat-prod.rst")
    pdb = app.PDBFile("last.pdb")
    system = ala2_solv.createSystem(
        nonbondedMethod=app.PME,
        rigidWater=True,
        switchDistance=1.0 * nanometer,
        nonbondedCutoff=1.2 * nanometer,
        constraints=app.HBonds,
    )
    # Create the integrator to do Langevin dynamics
    integrator = openmm.LangevinIntegrator(
        300 * kelvin,  # Temperature of heat bath
        1.0 / picoseconds,  # Friction coefficient
        1.0 * femtoseconds,  # Time step
    )

    # Define the platform to use; CUDA, OpenCL, CPU, or Reference. Or do not specify
    platform = Platform.getPlatformByName("CPU")
    # Create the Simulation object

    sim = app.Simulation(ala2_solv.topology, system, integrator, platform)

    # Set the particle positions
    #    sim.context.setPositions(ala2_solv.positions)
    sim.context.setPositions(pdb.getPositions(frame=-1))
    sim.reporters.append(app.PDBReporter("output.pdb", 200000))
    sim.reporters.append(app.DCDReporter("output.dcd", 200000))
    sim.reporters.append(StateDataReporter("data.txt", 20000, step=True, separator=" "))

    # Minimize the energy
    #    print('Minimizing energy')
    #    sim.minimizeEnergy()
    return sim


# %%
# functions for ploting and storing data
# functions for ploting and storing data
def plot_energy(result, plot_grid):

    energy = result["metapotential"]
    mesh = (compute_mesh(plot_grid) + 1) / 2 * plot_grid.size + plot_grid.lower
    print(mesh.shape)
    alpha = -(5.3 / (5.0))
    A = energy(mesh) * alpha
    surface = (A - A.min()).reshape(plot_grid.shape)

    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Free energy $[\\epsilon]$")

    ax.plot(mesh, surface, color="teal")

    fig.savefig("energy.png")


def save_energy_forces(result, plot_grid):
    energy = result["metapotential"]
    mesh = (compute_mesh(plot_grid) + 1) / 2 * plot_grid.size + plot_grid.lower
    print(mesh.shape)
    alpha = -(5.3 / (5.0))
    A = energy(mesh) * alpha
    surface = (A - A.min()).reshape(plot_grid.shape)
    numpy.savetxt("FES.csv", numpy.column_stack([mesh, surface]))


# %%
# %%
# %%
host = list(range(0, 144))
ligand = list(range(144, 168))
weights_host = np.ones(len(host)) / len(host)
weights_lig = np.ones(len(ligand)) / len(ligand)
anchor = 89
indices_sys = [ligand, host, [anchor]]
A = [4.6879, 4.8335, 1.12045]
B = [4.7829, 5.7927, 2.8729]
# anchor = 89
box = [5.5822, 5.5095, 5.4335]
Z_0 = 0.610225
Zcc = 1.2
R_cyl = 0.6
k_cone = 10000.0
k_cv = 20000.0
cv_min = 0.0
cv_max = 2.0
cv_buffer = 0.05
coordinates = open("referencesh.pdb", "r")
ref_loop1 = []
for line in coordinates:
    lista = line.split()
    id = lista[0]
    if id == "ATOM":
        atomid = int(lista[1])
        residue = int(lista[4])
        position = lista[5:8]
        temp_pos = []
        for p in position:
            temp_pos.append(float(p) / 10.0)
        ref_loop1.append(temp_pos)
coordinates.close()
# print(ref_loop1)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# print(indices_sys)
cvs = (
    Projection_on_Axis_mobile(
        indices_sys,
        references=ref_loop1,
        weights_lig=None,
        weights_prot=None,
        A=A,
        B=B,
        box=box,
    ),
)

grid = pysages.Grid(lower=(cv_min,), upper=(cv_max,), shape=(32,), periodic=False)

restraints = CVRestraints(lower=(-0.2,), upper=(2.0,), kl=(10000.0,), ku=(10000.0,))
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
funnel_force = get_funnel_force(
    indices_sys, ref_loop1, A, B, Zcc, Z_0, R_cyl, k_cone, k_cv, cv_min, cv_max, cv_buffer, box
)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
height = 0.2
sigma = 0.02
stride = 100
timesteps = 500
ngauss = timesteps // stride + 1
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methodo = Funnel_Metadynamics(
    cvs, height, sigma, stride, ngauss, deltaT=5000, kB=kB, external_force=funnel_force
)
funnel_file = "funnel.dat"
callback = Funnel_MetadLogger(funnel_file, 10)
sampling_context = SamplingContext(methodo, generate_simulation, callback)
# la linea de abajo se comenta para cuando se carga el pickle
state = pysages.run(sampling_context, timesteps)
# estas lineas se descomentan para usar el pickle para continuar la simulacion
# with open("restartb.pickle", "rb") as f:
#        state = pickle.load(f)
# state = pysages.run(state, generate_simulation, timesteps)
# se le puede cambiar e nombre del restart para no sobreescribirlo
with open("restartb.pickle", "wb") as f:
    pickle.dump(state, f)
result = pysages.analyze(state)
plot_energy(result, grid)
save_energy_forces(result, grid)

# %%
# %%
