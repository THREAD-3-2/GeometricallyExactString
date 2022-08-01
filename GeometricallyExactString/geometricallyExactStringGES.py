"""
Geometrically exact string.
"""

import jax.numpy as jnp
from jax import jit, jacrev, jacfwd, vmap

from typing import NamedTuple

# configurate JAX
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', True)

__all__ = ['GES']


class GES(NamedTuple):

    # Geometrically exact string
    # space
    nsteps: int             # number of segments in space
    step: float             # segment size in space
    s: jnp.ndarray          # realtive node positions in space
    L: float                # length of the string

    # geometry, material and parameters
    rho: jnp.ndarray        # mass density
    E: jnp.ndarray          # Young's modulus
    A: jnp.ndarray          # effective area
    C: jnp.ndarray          # Stiffness
    M: jnp.ndarray          # Mass

    # initial conditions
    q_0: jnp.ndarray        # initial configuration of the string
    v_0: jnp.array          # initial continuous velocity of the string


    def initString(L, nsteps, rho, E, A, initialConditions):
        """Create a gemetrically exact string.

        Parameters
        ----------
        L : float
            length of the string.
        nsteps : int
            number of discretization steps in space.
        rho : jnp.ndarray
            density.
        E : jnp.ndarray
            material value for the compressible Neo-Hookean material.
        A : jnp.ndarray
            area of the crosssection of the string.
        initialConditions : function
            function defining the relative position of one node in space.

        Returns
        -------
        GES : GES
            geometrically exact string.

        """
        C = jnp.array([E*A])
        M = jnp.array([rho*A])

        # space nodes
        # calculte segment size
        step = L/nsteps

        # create vector with relative node positions in space
        # nsteps+1 are the node
        s_vec = jnp.linspace(0., L, nsteps+1)

        # get the initial configuration of the string
        initialConditionsString = vmap(initialConditions)
        q_0, v_0 = initialConditionsString(s_vec)

        return GES(nsteps, step, s_vec, L, rho,
                   E, A, C, M, q_0, v_0)

    # # Lobatto I
    @jit
    def internalPotentialEnergy(q_n, q_np1, C, step):
        """Define the discrete internal potential energy.

        Parameters
        ----------
        q_n : jnp.ndarray
            configuration at node n in space.
        q_np1 : float
            cinfiguration at node n+1 in space.
        C : jnp.ndarray
            stiffness of the string.
        step : float
            step size in space.

        Returns
        -------
        V_int: jnp.ndarray
            discrete internal potential energy.

        """
        # approximation of the derivative in space
        nu_d = (q_np1 - q_n)/step

        # uniaxial rubber-like
        # stretch as eucldiean norm of the derivative
        nu_norm = jnp.sqrt(nu_d[0]**2 + nu_d[1]**2 + nu_d[2]**2)
        return 0.5 * C * (nu_norm**2 - 2.*jnp.log(nu_norm) - 1.)

    @jit
    def kineticEnergy(q_k, q_kp1, M, time_step):
        """Define the discrete kinetic eergy.

        Parameters
        ----------
        q_k : jnp.ndarray
            configuration at node k in time.
        q_kp1 : jnp.ndarray
            configuration at node k+1 in time.
        M : jnp.ndarray
            mass density of the string.
        time_step : flaot
            time step length.

        Returns
        -------
        T : jnp.ndarray
            discrete kinetic energy.

        """
        # velocity
        v_k = (q_kp1 - q_k)/time_step

        return 0.5 * M * jnp.transpose(v_k) @ v_k

    @jit
    def externalPotentialEnergy(q_n, step, M, g_):
        """Discrete external potential energy.

        Parameters
        ----------
        q_n : jnp.ndarray
            discrete configuration at node n in space.
        step : float
            step size in space.
        M : jnp.ndarray
            mass density of the string.
        g_ : float
            gravitaional acceleration.

        Returns
        -------
        V_ext : jnp.ndarray
            discrete external potential energy.

        """
        return M * jnp.transpose(q_n) @ jnp.array([0., 0., g_])

    @jit
    def dynamicLagrangian(q_n_k, q_np1_k, q_n_kp1, q_np1_kp1,
                          C, M, g_, step, time_step):
        """Calculate the discrete Lagrangian of the string.

        Parameters
        ----------
        q_n_k : jnp.ndarray
            discrete configuration at node n in space and k in time.
        q_np1_k : jnp.ndarray
            discrete configuration at node n+1 in space and k in time.
        q_n_kp1 : jnp.ndarray
            discrete configuration at node n in space and k+1 in time.
        q_np1_kp1 : jnp.ndarray
            discrete configuration at node n+1 in space and k+1 in time.
        C : jnp.ndarray
            stiffness of crosssection.
        M : jnp.ndarray
            mass density of string.
        g_ : float
            graviational acceleration.
        step : float
            step size in space.
        time_step : float
            time step size.

        Returns
        -------
        L : jnp.ndarray
            discrete Lagrangian.

        """
        T = 0.5 * (GES.kineticEnergy(q_n_k, q_n_kp1, M, time_step)
                   + GES.kineticEnergy(q_np1_k, q_np1_kp1, M, time_step))
        V_int = 0.5 * (GES.internalPotentialEnergy(q_n_k, q_np1_k, C, step)
                       + GES.internalPotentialEnergy(q_n_kp1, q_np1_kp1,
                                                     C, step))
        V_ext = 0.25 * (GES.externalPotentialEnergy(q_n_k, step, M, g_)
                        + GES.externalPotentialEnergy(q_n_kp1, step, M, g_)
                        + GES.externalPotentialEnergy(q_np1_k, step, M, g_)
                        + GES.externalPotentialEnergy(q_np1_kp1, step, M, g_))
        return step*time_step*(T - V_int - V_ext)

    @jit
    def D1dynamicLagrangian(q_n_k, q_np1_k, q_n_kp1, q_np1_kp1,
                            C,  M, g_, step, time_step):
        """Differentiate the discrete Lagrangian w.r.t. the first argument.

        Parameters
        ----------
        q_n_k : jnp.ndarray
            discrete configuration at node n in space and k in time.
        q_np1_k : jnp.ndarray
            discrete configuration at node n+1 in space and k in time.
        q_n_kp1 : jnp.ndarray
            discrete configuration at node n in space and k+1 in time.
        q_np1_kp1 : jnp.ndarray
            discrete configuration at node n+1 in space and k+1 in time.
        C : jnp.ndarray
            stiffness of crosssection.
        M : jnp.ndarray
            mass density of string.
        g_ : float
            graviational acceleration.
        step : float
            step size in space.
        time_step : float
            time step size.

        Returns
        -------
        D_1L_d : jnp.ndarray
            Derivative of the discrete Lagrangian w.r.t the first argument.

        """
        return jacrev(GES.dynamicLagrangian,
                      0)(q_n_k, q_np1_k, q_n_kp1, q_np1_kp1,
                         C,  M, g_, step, time_step).flatten()

    @jit
    def D2dynamicLagrangian(q_nm1_k, q_n_k, q_nm1_kp1, q_n_kp1,
                            C,  M, g_, step, time_step):
        """Differentiate the discrete Lagrangian w.r.t. the sceond argument.

        Parameters
        ----------
        q_n_k : jnp.ndarray
            discrete configuration at node n in space and k in time.
        q_np1_k : jnp.ndarray
            discrete configuration at node n+1 in space and k in time.
        q_n_kp1 : jnp.ndarray
            discrete configuration at node n in space and k+1 in time.
        q_np1_kp1 : jnp.ndarray
            discrete configuration at node n+1 in space and k+1 in time.
        C : jnp.ndarray
            stiffness of crosssection.
        M : jnp.ndarray
            mass density of string.
        g_ : float
            graviational acceleration.
        step : float
            step size in space.
        time_step : float
            time step size.

        Returns
        -------
        D_2L_d : jnp.ndarray
            Derivative of the discrete Lagrangian w.r.t the second argument.

        """
        return jacrev(GES.dynamicLagrangian,
                      1)(q_nm1_k, q_n_k, q_nm1_kp1, q_n_kp1,
                         C,  M, g_, step, time_step).flatten()

    @jit
    def D3dynamicLagrangian(q_n_km1, q_np1_km1, q_n_k, q_np1_k,
                            C,  M, g_, step, time_step):
        """Differentiate the discrete Lagrangian w.r.t. the third argument.

        Parameters
        ----------
        q_n_k : jnp.ndarray
            discrete configuration at node n in space and k in time.
        q_np1_k : jnp.ndarray
            discrete configuration at node n+1 in space and k in time.
        q_n_kp1 : jnp.ndarray
            discrete configuration at node n in space and k+1 in time.
        q_np1_kp1 : jnp.ndarray
            discrete configuration at node n+1 in space and k+1 in time.
        C : jnp.ndarray
            stiffness of crosssection.
        M : jnp.ndarray
            mass density of string.
        g_ : float
            graviational acceleration.
        step : float
            step size in space.
        time_step : float
            time step size.

        Returns
        -------
        D_3L_d : jnp.ndarray
            Derivative of the discrete Lagrangian w.r.t the third argument.

        """
        return jacrev(GES.dynamicLagrangian,
                      2)(q_n_km1, q_np1_km1, q_n_k, q_np1_k,
                         C,  M, g_, step, time_step).flatten()

    @jit
    def D4dynamicLagrangian(q_nm1_km1, q_n_km1, q_nm1_k, q_n_k,
                            C,  M, g_, step, time_step):
        """Differentiate the discrete Lagrangian w.r.t. the forth argument.

        Parameters
        ----------
        q_n_k : jnp.ndarray
            discrete configuration at node n in space and k in time.
        q_np1_k : jnp.ndarray
            discrete configuration at node n+1 in space and k in time.
        q_n_kp1 : jnp.ndarray
            discrete configuration at node n in space and k+1 in time.
        q_np1_kp1 : jnp.ndarray
            discrete configuration at node n+1 in space and k+1 in time.
        C : jnp.ndarray
            stiffness of crosssection.
        M : jnp.ndarray
            mass density of string.
        g_ : float
            graviational acceleration.
        step : float
            step size in space.
        time_step : float
            time step size.

        Returns
        -------
        D_4L_d : jnp.ndarray
            Derivative of the discrete Lagrangian w.r.t the forth argument.

        """
        return jacrev(GES.dynamicLagrangian,
                      3)(q_nm1_km1, q_n_km1, q_nm1_k, q_n_k,
                         C,  M, g_, step, time_step).flatten()

    @jit
    def assembleSystem(x_kp1, x_k, x_km1, string, time_step, g_):
        """Assemble the equations for the dynamic simulation of the GES.

        Parameters
        ----------
        x_kp1 : jnp.ndarray
            vector of unknows at time step k+1.
        x_k : jnp.ndarray
            vector of unknows at time step k.
        x_km1 : jnp.ndarray
            vector of unknows at time step k-1.
        string : GES
            Geometrically exact string.
        time_step : float
            stime step size.
        g_ : float
            graviational acceleration.

        Returns
        -------
        del_eq: jnp.ndarray
            discrete Euler-Lagrange equations.

        """
        step = string.step
        C = string.C
        M = string.M
        nsteps = jnp.shape(x_kp1)[0]//3

        q_kp1 = jnp.reshape(x_kp1, (nsteps, 3))
        q_k = jnp.reshape(x_k, (nsteps, 3))
        q_km1 = jnp.reshape(x_km1, (nsteps, 3))
        # vectorize Lagrangian for multiple nodes
        D1dynamicLagrangianString = vmap(GES.D1dynamicLagrangian,
                                         in_axes=[0, 0, 0, 0, None, None,
                                                  None, None, None])
        D2dynamicLagrangianString = vmap(GES.D2dynamicLagrangian,
                                         in_axes=[0, 0, 0, 0, None, None,
                                                  None, None, None])
        D3dynamicLagrangianString = vmap(GES.D3dynamicLagrangian,
                                         in_axes=[0, 0, 0, 0, None, None,
                                                  None, None, None])
        D4dynamicLagrangianString = vmap(GES.D4dynamicLagrangian,
                                         in_axes=[0, 0, 0, 0, None, None,
                                                  None, None, None])

        # calculate the stresses and momenta
        stress_m = D1dynamicLagrangianString(q_k[1:len(q_k)-1],
                                             q_k[2:len(q_k)],
                                             q_kp1[1:len(q_k)-1],
                                             q_kp1[2:len(q_k)],
                                             C, M, g_, step, time_step)
        momenta_m = D2dynamicLagrangianString(q_k[:len(q_k)-1],
                                              q_k[1:len(q_k)],
                                              q_kp1[:len(q_k)-1],
                                              q_kp1[1:len(q_k)],
                                              C, M, g_, step, time_step)
        stress_p = D3dynamicLagrangianString(q_km1[1:len(q_k)-1],
                                             q_km1[2:len(q_k)],
                                             q_k[1:len(q_k)-1],
                                             q_k[2:len(q_k)],
                                             C, M, g_, step, time_step)
        momenta_p = D4dynamicLagrangianString(q_km1[:len(q_k)-1],
                                              q_km1[1:len(q_k)],
                                              q_k[:len(q_k)-1],
                                              q_k[1:len(q_k)],
                                              C, M, g_, step, time_step)

        del_eq = jnp.zeros((nsteps, 3))
        del_eq = del_eq.at[1:jnp.shape(del_eq)[0]-1, :].add(stress_m
                                                            + stress_p)
        del_eq = del_eq.at[1:jnp.shape(del_eq)[0], :].add(momenta_m
                                                          + momenta_p)

        return del_eq.flatten()

    @jit
    def jacobianAssembleSystem(q_kp1, q_k, q_km1, string, time_step, g_):
        """Calculate the jacobian of the DEL w.r.t. the conifugration at k+1.

        Parameters
        ----------
        q_kp1 : jnp.ndarrray
            configuration at node k+1 in time.
        q_k : jnp.ndarrray
            configuration at node k in time.
        q_km1 : jnp.ndarrray
            configuration at node k-1 in time.
        string : GES
            Geomtrically exact string.
        time_step : float
            time step size.
        g_ : float
            graviational acceleration.

        Returns
        -------
        jnp.ndarray
            jacobian of the DEL.

        """
        return jacfwd(GES.assembleSystem, 0)(q_kp1, q_k, q_km1,
                                             string, time_step, g_)

    @jit
    def assembleSystem1(x_kp1, x_k, v_0, string, time_step, g_):
        """Assemble the initial time step of the GES.

        Parameters
        ----------
        x_kp1 : jnp.ndarray
            vector of unknows at time step k+1.
        x_k : jnp.ndarray
            vector of unknows at time step k.
        v_0 : jnp.ndarray
            vector of initial velocities.
        string : GES
            Geometrically exact string.
        time_step : float
            stime step size.
        g_ : float
            graviational acceleration.

        Returns
        -------
        del_eq: jnp.ndarray
            discrete Euler-Lagrange equations.

        """
        step = string.step
        C = string.C
        M = string.M
        nsteps = jnp.shape(x_kp1)[0]//3

        q_kp1 = jnp.reshape(x_kp1, (nsteps, 3))
        q_k = jnp.reshape(x_k, (nsteps, 3))
        # vectorize Lagrangian for multiple nodes
        D1dynamicLagrangianString = vmap(GES.D1dynamicLagrangian,
                                         in_axes=[0, 0, 0, 0, None, None,
                                                  None, None, None])
        D2dynamicLagrangianString = vmap(GES.D2dynamicLagrangian,
                                         in_axes=[0, 0, 0, 0, None, None,
                                                  None, None, None])

        # calculate the stresses and momenta
        stress_m = D1dynamicLagrangianString(q_k[0:len(q_k)-1],
                                             q_k[1:len(q_k)],
                                             q_kp1[:len(q_k)-1],
                                             q_kp1[1:len(q_k)],
                                             C, M, g_, step, time_step)
        momenta_m = D2dynamicLagrangianString(q_k[:len(q_k)-1],
                                              q_k[1:len(q_k)],
                                              q_kp1[:len(q_k)-1],
                                              q_kp1[1:len(q_k)],
                                              C, M, g_, step, time_step)

        # first time step, derive momenta and stresses
        # from continuous legendre transformation
        p = M * v_0

        del_eq = jnp.zeros((nsteps, 3))
        del_eq = del_eq.at[1:jnp.shape(del_eq)[0], :].add(stress_m)
        del_eq = del_eq.at[1:jnp.shape(del_eq)[0], :].add(momenta_m)

        return (p + del_eq).flatten()

    @jit
    def jacobianAssembleSystem1(q_kp1, q_k, v_0, string, time_step, g_):
        """Calculate the jacobian of the DEL w.r.t. the conifugration at k+1.

        Parameters
        ----------
        q_kp1 : jnp.ndarrray
            configuration at node k+1 in time.
        q_k : jnp.ndarrray
            configuration at node k in time.
        v_0 : jnp.ndarrray
            initial velocity.
        string : GES
            Geomtrically exact string.
        time_step : float
            time step size.
        g_ : float
            graviational acceleration.

        Returns
        -------
        jnp.ndarray
            jacobian of the DEL.

        """
        return jacfwd(GES.assembleSystem,
                      0)(q_kp1, q_k, v_0, string, time_step, g_)
