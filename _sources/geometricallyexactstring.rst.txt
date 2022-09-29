.. _geometricallyexactstring:

Geometrically exact string model
================================

Continuous model
----------------

We consider the arc length of a stress-free reference configuration :math:`s \in [0, L] \subseteq \mathbb{R}` and time :math:`t \in [0, T] \subseteq \mathbb{R}`.
The deformation of the string is then described by :math:`r(s, t): [0, L] \times [0, T] \mapsto \mathbb{R}^3`. Its velocity is :math:`v(s,t) = \frac{\partial r(s, t)}{\partial t}`.
The stretch is defined as :math:`\nu(s,t) = \left\Vert \frac{\partial r(s,t)}{\partial s} \right\Vert`.

The kinetic energy density is :math:`T(v) = \frac{1}{2} v^T \rho A v` with the mass density :math:`\rho` and the area of the cross-section in reference configuration :math:`A`.

.. autofunction:: GeometricallyExactString.geometricallyExactStringGES.kineticEnergy

The inner potential energy assumes a compressible Neo-Hookean material model that is adapted to the string kinematics thus only the tangential stretch is considered. The determinant of the deformation gradient is therefore the stretch.
The inner potential energy density is :math:`U_{int}(\nu) =  \frac{1}{2} C (\nu^2 - \text{ln}(\nu) - 1)` with the material constant :math:`C`.

.. autofunction:: GeometricallyExactString.geometricallyExactStringGES.internalPotentialEnergy

The external potential energy density only considers gravity in this code, :math:`U_{ext}(r) =  \rho A g r` with the gravitational acceleration vector :math:`g`.

.. autofunction:: GeometricallyExactString.geometricallyExactStringGES.externalPotentialEnergy

The Lagranian density is thus :math:`L(r, v, \nu) = T(v) - U_{int}(\nu) - U_{ext}(r)`. Only fixed-free boundary conditions in space are considered.

.. autofunction:: GeometricallyExactString.geometricallyExactStringGES.dynamicLagrangian

Discrete Model
--------------

The action

.. math::
    \begin{aligned}
        &\int_{t_{n}}^{t_{n+1}} \int_{s_{n}}^{s_{n+1}} L(r, v, \nu) ds dt \approx \frac{\Delta t \Delta s}{4} \Bigg(&&L_d(r_{n}^{k}, \frac{r_{n}^{k+1} - r_{n}^{k}}{\Delta t}, \frac{r_{n+1}^{k} - r_{n}^{k}}{\Delta s}) \\
        & &&+ L_d(r_{n+1}^{k}, \frac{r_{n}^{k+1} - r_{n}^{k}}{\Delta t}, \frac{r_{n+1}^{k} - r_{n}^{k}}{\Delta s})\\
        & &&+ L_d(r_{n}^{k+1}, \frac{r_{n}^{k+1} - r_{n}^{k}}{\Delta t}, \frac{r_{n+1}^{k+1} - r_{n}^{k+1}}{\Delta s})\\
        & &&+ L_d(r_{n+1}^{k+1}, \frac{r_{n+1}^{k+1} - r_{n+1}^{k}}{\Delta t}, \frac{r_{n+1}^{k+1} - r_{n}^{k+1}}{\Delta s}) \Bigg)
    \end{aligned}
.. math::

is approximated by the trapezoidal rule in space and time. Further explanations on this method, applied to the 1D wave equation can be found `here <https://thread-3-2.github.io/1D_wave_equation/discretisation.html>`_. Mind, that the string under consideration has hyperelastic material and gravitation is also considered. The discrete variation of this discrete approximation of the action leads to the discrete Euler-Lagrange field equations for the geometrically exact string. The discrete variations are calculated in 

.. autofunction:: GeometricallyExactString.geometricallyExactStringGES.D1dynamicLagrangian

as well as in other functions for different nodes in space and time. The equations of the string with its hard coded boundary conditions are assembled in:

.. autofunction:: GeometricallyExactString.assembleSystem

The jacobian of those equations is required for the efficient solution of these equations. It is derived via automatic differentiation in:

.. autofunction:: GeometricallyExactString.jacobianAssembleSystem

The latter two function also have versions for the first time step that use the Legendre transformation to apply consistent initial conditions for position and velocity.

Time stepping
-------------

The time stepping routine is

.. autofunction:: GeometricallyExactString.simulate
