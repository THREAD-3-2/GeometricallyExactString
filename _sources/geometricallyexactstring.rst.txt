.. _geometricallyexactstring:

========================
Geometrically exact string model
========================

Continuous model
------------------------------

We consider the arc length of a stress-free reference configuration :math:`s \in [0, L] \subseteq \mathbb{R}` and time :math:`t \in [0, T] \subseteq \mathbb{R}`.
The deformation of the string is then described by :math:`r(s, t): [0, L] \times [0, T] \mapsto \mathbb{R}^3`. Its velocity is :math:`v(s,t) = \frac{\partial r(s, t)}{\partial t}`.
The stretch is defined as :math:`\nu(s,t) = \left\Vert \frac{\partial r(s,t)}{\partial s} \right\Vert`.

The kinetic energy density is :math:`T(v) = \frac{1}{2} v^T \rho A v` with the mass density :math:`\rho` and the area of the cross-section in reference configuration :math:`A`.
The inner potential energy assumes a compressible Neo-Hookean material model that is adapted to the string kinematics thus only the tangential stretch is considered. The determinant of the deformation gradient is therefore the stretch.
The inner potential energy density is :math:`U_{int}(\nu) =  \frac{1}{2} C (\nu^2 - \text{ln}(\nu) - 1)` with the material constant :math:`C`. The external potential energy density only considers gravity in this code, :math:`U_{ext}(r) =  \rho A g r` with the gravitational acceleration vector :math:`g`.
The Lagranian density is thus :math:`L(r, v, \nu) = T(v) - U_{int}(\nu) - U_{ext}(r)`. Only fixed-free boundary conditions in space are considered.

Discrete Model
-------------------------------------------------

The action

.. math::
    \int_{t_{n}}^{t_{n+1}} L(r, v, \nu) dt \approx \frac{\Delta t \Delta s}{4}(L_d(r_{n}^{k}, \frac{r_{n}^{k+1} - r_{n}^{k}}{\Delta t}, \frac{r_{n+1}^{k} - r_{n}^{k}}{\Delta s}) + L_d(r_{n+1}^{k}, \frac{r_{n}^{k+1} - r_{n}^{k}}{\Delta t}, \frac{r_{n+1}^{k} - r_{n}^{k}}{\Delta s}) + L_d(r_{n}^{k+1}, \frac{r_{n}^{k+1} - r_{n}^{k}}{\Delta t}, \frac{r_{n+1}^{k+1} - r_{n}^{k+1}}{\Delta s}) + L_d(r_{n+1}^{k+1}, \frac{r_{n+1}^{k+1} - r_{n+1}^{k}}{\Delta t}, \frac{r_{n+1}^{k+1} - r_{n}^{k+1}}{\Delta s}) )
.. math::

is approximated by the trapezoidal rule in space and time. The discrete variation of this discrete approximation of the action leads to the discrete Euler-Lagrange field equations for the geometrically exact string.

Time stepping
-------------------------------------------------

The time stepping routine is
.. autofunction:: GeometricallyExactString.plotTipTrajectory