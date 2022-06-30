.. _example:

=========
 Example
=========

This example shows how to simulate the evolution of a simple pendulum
with the symplectic Euler method.


Problem description
===================

The state space is :math:`\mathbb{R}^2`
and the initial value problem
whose solution is the evolution of the pendulum dynamics is

.. math::

    \begin{aligned}
        \dot{q} &= p \\
        \dot{p} &= -\sin(q) \\
        q(0) &= 1 \\
        p(0) &= 0
        \,.
    \end{aligned}


.. note::

    The right hand side of the first two equations define a vector field on :math:`\mathbb{R}^2`
    and the last two equations define a point on :math:`\mathbb{R}^2`.


Simulation
==========

First, the vector field is defined as a Python function,
which in turn is used to define the :ref:`initial value problem (IVP) <ivp>`,
Then, the energy function (Hamiltonian) of the pendulum is defined as a Python function.
The IVP is :ref:`solved <simulate>` with the symplectic Euler method
using a time step of 0.25 seconds for a duration of 55 seconds,
Finally, the trajectory, the phase portrait and the evolution of the energy are plotted,
see :ref:`plotting`.

.. doctest::

    >>> import autograd.numpy as np
    >>> from example_gni_project import *

    >>> def pendulum_vector_field(x):
    ...     q, p = x
    ...     return np.array([p, -np.sin(q)])

    >>> pendulum_ivp = IVP(
    ...     ("q", "p"),
    ...     pendulum_vector_field,
    ...     np.array([1., 0.])
    ... )

    >>> def pendulum_energy(x):
    ...     q, p = x
    ...     return 1/2 * p**2 + (1 - np.cos(q))

    >>> simulation = simulate(
    ...     pendulum_ivp,
    ...     symplectic_euler,
    ...     0.25,
    ...     duration=55.0
    ... )

    >>> plot_trajectory(simulation)
    >>> plot_phase_portrait(simulation)
    >>> plot_observable(simulation, "energy", pendulum_energy)

