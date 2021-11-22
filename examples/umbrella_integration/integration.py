#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import hoomd
hoomd.context.initialize()
import hoomd.md as md
import hoomd.dlext

import pysages
from pysages.collective_variables import Component
from pysages.methods import UmbrellaIntegration


param1 = {"A": 1., "w": 0.2, "p": 2}

def generate_context(**kwargs):
    context = hoomd.context.SimulationContext()
    with context:
        print("Operating replica {0}".format(kwargs.get("replica_num")))
        system = hoomd.init.read_gsd("start.gsd")

        hoomd.md.integrate.nve(group=hoomd.group.all())
        hoomd.md.integrate.mode_standard(dt=0.01)

        nl = hoomd.md.nlist.cell()
        dpd = hoomd.md.pair.dpd(r_cut=1, nlist=nl, seed=42, kT=1.)
        dpd.pair_coeff.set("A", "A", A=5., gamma=1.0)
        dpd.pair_coeff.set("A", "B", A=5., gamma=1.0)
        dpd.pair_coeff.set("B", "B", A=5., gamma=1.0)

        periodic = hoomd.md.external.periodic()
        periodic.force_coeff.set('A', A=param1["A"], i=0, w=param1["w"], p=param1["p"])
        periodic.force_coeff.set('B', A=0.0, i=0, w=0.02, p=1)
    return context


def plot_hist(result, bins=35):
    fig, ax = plt.subplots(3, 3)

    # ax.set_xlabel("CV")
    # ax.set_ylabel("p(CV)")

    counter = 0
    hist_per = len(result["center"])//9+1
    for x in range(3):
        for y in range(3):
            for i in range(hist_per):
                if counter+i < len(result["center"]):
                    center = np.asarray(result["center"][counter+i])
                    histo, edges = result["histogram"][counter+i].get_histograms(bins=bins)
                    edges = np.asarray(edges)[0]
                    edges = (edges[1:] + edges[:-1]) / 2
                    ax[x,y].plot(edges, histo, label="center {0}".format(center))
                    ax[x,y].legend(loc="best")
            counter += hist_per
    while counter < len(result["center"]):
        center = np.asarray(result["center"][counter])
        histo, edges = result["histogram"][counter].get_histograms(bins=bins)
        edges = np.asarray(edges)[0]
        edges = (edges[1:] + edges[:-1]) / 2
        ax[2,2].plot(edges, histo, label="center {0}".format(center))
        ax[2,2].legend(loc="best")
        counter += 1

    fig.savefig("hist.pdf")

def external_field(r, A, p, w):
    return A*np.tanh(1/(2*np.pi*p*w)*np.cos(p*r))

def plot_energy(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Free energy $[\epsilon]$")
    center = np.asarray(result["center"])
    A = np.asarray(result["A"])
    offset = np.min(A)
    ax.plot(center, A+offset, color="teal")

    x = np.linspace(-3, 3, 50)
    data = external_field(x, **param1)
    offset = np.min(data)
    ax.plot(x, data+offset, label="test")

    fig.savefig("energy.pdf")


def main():
    cvs = [Component([0], 0),]
    method = UmbrellaIntegration(cvs)

    k = 15.
    Nreplica = 25
    centers = list(np.linspace(-1.5, 1.5, Nreplica))
    result = method.run(generate_context, int(1e5), centers, k, 10)

    plot_energy(result)
    plot_hist(result)


if __name__ == "__main__":
    main()
