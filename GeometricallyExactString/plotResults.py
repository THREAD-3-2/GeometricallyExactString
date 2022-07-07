#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot simualtion results
"""

import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# configurate JAX
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', True)

__all__ = ['plotTipTrajectory', 'createAnimation']

def plotTipTrajectroy(x, T, nt):
    """Plot the x-position of the tip of the string.

    Parameters
    ----------
    x : jnp.ndarray
        solution trajectory.
    T : float
        simulation time.
    nt : int
        number of time steps.

    Returns
    -------
    None.

    """
    fig = plt.figure(constrained_layout=True)
    t = jnp.linspace(0, T, nt)
    plt.plot(t, x[len(x)-1, :])
    plt.grid(color='0.9')
    plt.xlabel('time t')
    plt.ylabel('x position of the tip of the string')
    fig = plt.gcf()
    fig.savefig('error energy density vs time t')


def createAnimation(x, string, T, nt):
    """Create an animation of the y-z plane.

    Parameters
    ----------
    x : jnp.ndarray
        positions of all nodes of the string over time.
    string : GES
        Geometrically exact string.
    T : float
        simulation time.
    nt : int
        number of time steps.

    Returns
    -------
    anim : d
        Animation of the string.

    """
    x_an = jnp.reshape(x, (string.nsteps+1, 3, nt))
    # create animation
    fig = plt.figure()
    ax = plt.axes(xlim=(1.1 * jnp.amin(x_an[:, 1, :]),
                        1.1 * jnp.amax(x_an[:, 1, :])),
                  ylim=(1.1 * jnp.amin(x_an[:, 2, :]),
                        1.1 * jnp.amax(x_an[:, 2, :])))
    ax.set_aspect('equal')
    line, = ax.plot([], [], lw=3)
    plt.grid(color='0.9')

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x_ani = x_an[:, 1, 25 * i]
        y_ani = x_an[:, 2, 25 * i]
        line.set_data(x_ani, y_ani)
        return line,

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=nt//25,
                                   interval=(nt/25)/T,
                                   blit=False)

    anim.save('string.gif', writer='imagemagick')
