.. _example:

=========
 Example
=========

This example shows how to simulate a geometrically exact string with fixed-free boundary conditions.


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

    >>> from GeometricallyExactString import simulateString
    >>> from GeometricallyExactString import plotResults
    >>> from geometricallyExactString import GES
    >>> import jax.numpy as jnp

    >>> T = 1.     # simulation time
    >>> nt = 500   # number of time steps
    >>> dt = T/nt  # time step width
    >>> g_ = 9.81  # graviational constant

    >>> def initialConditionsString(s):
    ...     return jnp.array([0., 0., -1.5*s]), jnp.array([0., 0.1*s, 0.])
    
    >>> string = GES.initString(L=10, nsteps=25, rho=4.5e0, E=1e2, A=0.1,
    ...                         initialConditions=initialConditionsString)

    >>> x = simulateString.simulate(string, nt, dt, g_)

    >>> plotResults.plotTipTrajectroy(x, T, nt)
    >>> plotResults.createAnimation(x, string, T, nt)

