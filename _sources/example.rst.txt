.. _example:

Example
=======

This example shows how to simulate a geometrically exact string with fixed-free boundary conditions.


Problem description
-------------------

The string is fixed at one end and free at the other. It hangs down under gravity and has linearly increasing velocity over its length.


Simulation
----------

First, the initial conditions are defined as a Python function. The relative position :math:`s \in [0, L]`
defines the node location along the string. Here, the string is initially stretched by :math:`50 \%` and has a linearly increasing velocity.
Then, a string is initialized, with the desired length :math:`L`, number of discretization steps :math:`n_s`,
mass density :math:`\rho` and material parameter :math:`E` as well as the area of the cross-section :math:`A`.

.. autofunction:: GeometricallyExactString.initString

The dynamic simulation is then started and the results plotted.

.. doctest::

    >>> from GeometricallyExactString import *
    >>> import jax.numpy as jnp

    >>> T = 10.     # simulation time
    >>> nt = 2500   # number of time steps
    >>> dt = T/nt  # time step width
    >>> g_ = 9.81  # graviational constant

    >>> def initialConditionsString(s):
    ...     return jnp.array([0., 0., -1.5*s]), jnp.array([0., 0.1*s, 0.])
    
    >>> string = initString(L=10, nsteps=25, rho=4.5e0, E=1e2, A=0.1,
    ...                         initialConditions=initialConditionsString)

    >>> x = simulate(string, nt, dt, g_)

    >>> plotTipTrajectory(x, T, nt)
    >>> createAnimation(x, string, T, nt)

This example results in the following plot and animation:

.. image:: z_position_vs_time_t.png

It may take a while to render the animation:

.. image:: string.gif
