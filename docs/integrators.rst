.. _integrators:

=====================
 Integration methods
=====================

.. _simulate:

Generic time-stepping routine
-----------------------------

All integration methods are used via the same time-stepping routine:

.. autofunction:: example_gni_project.simulate


.. _nonsymplectic:

Non-symplectic methods
======================

Below
:math:`X \colon \mathcal{X} \rightarrow T \mathcal{X}` denotes the vector field
and :math:`x_0 \in \mathcal{X}` denotes the initial state.
Together they define the initial value problem.
Further, :math:`x_1 \in \mathcal{X}` denotes the state of the system
after one time step of size :math:`h`.


`explicit_euler`
----------------

The explicit Euler method is the simplest approximation one can make:

.. math::

    x_1 = x_0 + h \, X(x_0)


The method is first-order
and unstable in the sense that
the integrator will predict a gain of energy due to numerical errors.



`implicit_euler`
----------------

The implicit Euler method is already significantly more complicated in its implementation
since it requires solving a system of possibly nonlinear equations:

.. math::

    x_1 = x_0 + h \, X(x_1)


The method is first-order
and stable in the sense that
the integrator will predict a loss of energy due to numerical errors.


.. _symplectic:

Symplectic methods
===================

Symplectic methods have good long-time energy behavior.
They (almost) conserve the energy (Hamiltonian) of a :ref:`Hamiltonian system <hamiltonian_ivp>`.
More precisely, symplectic methods do not conserve the physical energy
but they conserve a nearby/perturbed Hamiltonian,
see e.g. `Reich 1999 <https://doi.org/10.1137/S0036142997329797>`_ and references therein.

Since symplectic methods mimic a (canonical) Hamiltonian structure at the discrete level,
the state is partitioned into
configuration variables :math:`q` and conjugate momentum variables :math:`p`,
i.e. :math:`x = \left[ q, \, p \right]`.


`symplectic_euler`
------------------

The symplectic Euler method essentially combines
the explicit Euler method and the implicit Euler method:

.. math::

    \begin{bmatrix}
        q_1 \\
        p_1
    \end{bmatrix}
    =
    \begin{bmatrix}
        q_0 \\
        p_0
    \end{bmatrix}
    + h \,
    X \biggl(
    \begin{bmatrix}
        q_0 \\
        p_1
    \end{bmatrix}
    \biggr)


`symplectic_euler2`
-------------------

Another symplectic Euler method results when combining
the implicit Euler method and the explicit Euler method
in the other obvious way:

.. math::

    \begin{bmatrix}
        q_1 \\
        p_1
    \end{bmatrix}
    =
    \begin{bmatrix}
        q_0 \\
        p_0
    \end{bmatrix}
    + h \,
    X \biggl(
    \begin{bmatrix}
        q_1 \\
        p_0
    \end{bmatrix}
    \biggr)


`störmer_verlet`
----------------

The Störmer-Verlet method results from the composition of
the first and the second symplectic Euler method,
which is asserted by the `test_composition_störmer_verlet` function
in `tests/test_integrators.py`.
To concretely describe the Störmer-Verlet method,
the vector field is partitioned analogously to the state
i.e.
:math:`x = \left[ q, \, p \right]` and
:math:`X = \left[ X_q, \, X_p \right] = \left[ +\frac{\partial H(q, p)}{\partial p}, -\frac{\partial H(q, p)}{\partial q} \right]`.
As long as :math:`H` is separable,
:math:`X_q` only depends on :math:`p`
while
:math:`X_p` only depends on :math:`q`.


.. math::

   \begin{align}
        p_{1/2} &= p_0 + \frac{h}{2} \, X_p(q_0) \\
        q_1 &= q_0 + h \, p_{1/2} \\
        p_1 &= p_{1/2} + \frac{h}{2} \, X_p(q_1)
    \end{align}


The Störmer-Verlet method is the composition of the two symplectic Euler methods.
As such, it is a symmetric second-order method.
The symmetry is asserted by `test_symmetry_störmer_verlet` in `test/test_integrators.py`.
