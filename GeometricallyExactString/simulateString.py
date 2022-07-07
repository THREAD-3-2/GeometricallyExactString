#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate the string
"""

import jax.numpy as jnp
from geometricallyExactString import GES

from scipy import optimize
from tqdm import tqdm

# configurate JAX
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', True)


def simulate(string, nt, dt, g_):
    """Simulate the string.

    Parameters
    ----------
    string : GES
        geometrically exact string.
    nt : int
        number of time steps.
    dt : float
        time step size.
    g_ : float
        gravitational acceleration.

    Returns
    -------
    x : jnp.ndarray
        solution of the dynamic simulation

    """
    # create solution vector
    x = jnp.zeros((3*(string.nsteps+1), nt))
    # set initial position
    x = x.at[:, 0].set(string.q_0.flatten())

    # first time step
    sol_dyn = optimize.root(GES.assembleSystem1,
                            x[:, 0],
                            args=(x[:, 0], string.v_0, string, dt, g_),
                            method='hybr',
                            jac=GES.jacobianAssembleSystem1,
                            options={'xtol': 1e-12})
    x = x.at[:, 1].set(sol_dyn.x)

    # solve all other time steps
    for n_t in tqdm(range(1, nt+1)):
        sol_dyn = optimize.root(GES.assembleSystem,
                                x[:, n_t],
                                args=(x[:, n_t], x[:, n_t-1], string, dt, g_),
                                method='hybr',
                                jac=GES.jacobianAssembleSystem,
                                options={'xtol': 1e-12})
        x = x.at[:, n_t+1].set(sol_dyn.x)

    return x
