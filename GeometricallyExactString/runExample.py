#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:42:11 2022

@author: Matthias Schubert
"""

import simulateString
import plotResults
from geometricallyExactStringGES import GES
import jax.numpy as jnp

if __name__ == "__main__":

    # simulation parameters
    T = 10.     # simulation time
    nt = 5000    # number of time steps
    dt = T/nt  # time step width
    g_ = 9.81 # graviational constant

    def initialConditionsString(s):
        # define the initial configuration q_0 and velocity v_0 of the string
        # string is fixed at s=0, thus v_0(s=0) = 0 must hold
        return jnp.array([0., 0., -1.5*s]), jnp.array([0., 0.1*s, 0.])

    # create string
    string = GES.initString(L=10, nsteps=25, rho=4.5, E=1e2, A=0.1,
                            initialConditions=initialConditionsString)

    x = simulateString.simulate(string, nt, dt, g_)

    plotResults.plotTipTrajectory(x, T, nt)
    plotResults.createAnimation(x, string, T, nt)
