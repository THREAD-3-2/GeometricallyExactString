.. _example:

=========
 Example
=========

This example shows how to simulate a geometrically exact string with fixed-free boundary conditions.


Problem description
===================

The string is fixed at one end and free at the other end. It hangs down under gravity. 


Simulation
==========

First, the initial conditions are defined as a Python function. The relative position :math:`s\in\[0, L\]`
defines the node location along the string. Here, the string is initially stretched by :math:`50\%` with linearly increasing velocity in y direction.
Then, a string is initialized, with the desired length :math:`L`, number of discretization steps :math:`n_s`,
mass density :math:`\rho` and material parameter :math:`E` as well as the area of the cross-section :math:`A`.
The dynamic simulation is then started and the results plotted.

.. doctest::

    >>> from GeometricallyExactString import *
    >>> import jax.numpy as jnp

    >>> T = 10.     # simulation time
    >>> nt = 5000   # number of time steps
    >>> dt = T/nt  # time step width
    >>> g_ = 9.81  # graviational constant

    >>> def initialConditionsString(s):
    ...     return jnp.array([0., 0., -1.5*s]), jnp.array([0., 0.1*s, 0.])
    
    >>> string = GES.initString(L=10, nsteps=25, rho=4.5e0, E=1e2, A=0.1,
    ...                         initialConditions=initialConditionsString)

    >>> x = simulateString.simulate(string, nt, dt, g_)

    >>> plotResults.plotTipTrajectroy(x, T, nt)
    >>> plotResults.createAnimation(x, string, T, nt)

