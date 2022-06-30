#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:00:21 2022

@author: MSchubert
"""

import jax.numpy as jnp
from jax import jit, jacrev, jacfwd, vmap

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy import optimize
from tqdm import tqdm
from typing import NamedTuple 
from functools import partial

# configurate JAX
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_x64', True)

@jit 
def intiialConditionsString(s):
    # define the initial configuration q_0 and velocity v_0 of the string
    # string is fixed at s=0, thus v_0(s=0) = 0
    return jnp.array([0., 0., -1.5*s]), jnp.array([0., 0.1*s**3, 0.])

class GeometricallyExactString(NamedTuple):
    # Geometrically exact string
    # space
    nsteps: int             # number of segments in space, number of nodes is nsteps+1
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
        # Initilize a geometrically exact string
        
        C = jnp.array([E*A])
        M = jnp.array([rho*A])

        ## space nodes
        # calculte segment size
        step = L/nsteps
        
        # create vector with relative node positions in space, nsteps+1 are the node 
        s_vec = jnp.linspace(0., L, nsteps+1)
        
        # get the initial configuration of the string
        initialConditionsString = vmap(initialConditions)
        q_0, v_0 = initialConditionsString(s_vec)
           
        return GeometricallyExactString(nsteps, step, s_vec, L, rho,
                                        E, A, C, M, q_0, v_0)
    
    # # Lobatto I
    @jit
    def internalPotentialEnergy(q_n, q_np1, C, step):
        # discrete internal potential energy of the string
        
        # approximation of the derivative in space
        nu_d = (q_np1 - q_n)/step
           
        # uniaxial rubber-like
        # stretch as eucldiean norm of the derivative
        nu_norm = jnp.sqrt(nu_d[0]**2 + nu_d[1]**2 + nu_d[2]**2)
        
        # return 0.5 * C * (nu_norm**2 - 2.*jnp.log(nu_norm) - 1.) 
        return 0.5 * C * (nu_norm**2 - 2.*jnp.log(nu_norm) - 1.)

    @jit
    def kineticEnergy(q_k, q_kp1, M, time_step):
        # discrete kinetic energy of a node of a string
        
        # velocity
        v_k = (q_kp1 - q_k)/time_step
         
        return 0.5 * M * jnp.transpose(v_k) @ v_k 
    
    @jit 
    def dynamicLagrangian(q_n_k, q_np1_k, q_n_kp1, q_np1_kp1, C,  M, step, time_step):
        # dynamic Lagrangian for two nodes in space and time each
        # internal potential energy does not depend on configuration, only its derivative
        T =  0.5 * (GeometricallyExactString.kineticEnergy(q_n_k, q_n_kp1, M, time_step)
                                + GeometricallyExactString.kineticEnergy(q_np1_k, q_np1_kp1, M, time_step))
        V_int = 0.5 * (GeometricallyExactString.internalPotentialEnergy(q_n_k, q_np1_k, C, step) 
                             + GeometricallyExactString.internalPotentialEnergy(q_n_kp1, q_np1_kp1, C, step))    
        return step*time_step*(T - V_int)
    
    @jit 
    def D1dynamicLagrangian(q_n_k, q_np1_k, q_n_kp1, q_np1_kp1, C,  M, step, time_step):
        return jacrev(GeometricallyExactString.dynamicLagrangian
                      , 0)(q_n_k,q_np1_k, q_n_kp1, q_np1_kp1, C,  M, step, time_step).flatten()
    
    @jit 
    def D2dynamicLagrangian(q_nm1_k, q_n_k, q_nm1_kp1, q_n_kp1, C,  M, step, time_step):
        return jacrev(GeometricallyExactString.dynamicLagrangian
                      , 1)(q_nm1_k, q_n_k, q_nm1_kp1, q_n_kp1, C,  M, step, time_step).flatten()
    
    @jit 
    def D3dynamicLagrangian(q_n_km1, q_np1_km1, q_n_k, q_np1_k, C,  M, step, time_step):
        return jacrev(GeometricallyExactString.dynamicLagrangian
                      , 2)(q_n_km1, q_np1_km1, q_n_k, q_np1_k, C,  M, step, time_step).flatten()
    
    @jit 
    def D4dynamicLagrangian(q_nm1_km1, q_n_km1, q_nm1_k, q_n_k, C,  M, step, time_step):
        return jacrev(GeometricallyExactString.dynamicLagrangian
                      , 3)(q_nm1_km1, q_n_km1, q_nm1_k, q_n_k, C,  M, step, time_step).flatten()

    @jit
    def assembleSystem(x_kp1, x_k, x_km1, string, time_step):
        # assemble the discrete Euler-Lagrange field equations
        
        step = string.step
        C = string.C
        M = string.M
        nsteps = jnp.shape(x_kp1)[0]//3
        
        q_kp1 = jnp.reshape(x_kp1, (nsteps, 3))
        q_k = jnp.reshape(x_k, (nsteps, 3))
        q_km1 = jnp.reshape(x_km1, (nsteps, 3))
        # vectorize Lagrangian for multiple nodes
        D1dynamicLagrangianString = vmap(GeometricallyExactString.D1dynamicLagrangian, in_axes=[0, 0, 0, 0, None, None, None, None])
        D2dynamicLagrangianString = vmap(GeometricallyExactString.D2dynamicLagrangian, in_axes=[0, 0, 0, 0, None, None, None, None])
        D3dynamicLagrangianString = vmap(GeometricallyExactString.D3dynamicLagrangian, in_axes=[0, 0, 0, 0, None, None, None, None])
        D4dynamicLagrangianString = vmap(GeometricallyExactString.D4dynamicLagrangian, in_axes=[0, 0, 0, 0, None, None, None, None])
        
        # calculate the stresses and momenta
        stress_m = D1dynamicLagrangianString(q_k[1:len(q_k)-1], q_k[2:len(q_k)],
                                                      q_kp1[1:len(q_k)-1], q_kp1[2:len(q_k)],
                                                      C, M, step, time_step)
        momenta_m = D2dynamicLagrangianString(q_k[:len(q_k)-1], q_k[1:len(q_k)],
                                                      q_kp1[:len(q_k)-1], q_kp1[1:len(q_k)],
                                                      C, M, step, time_step)
        stress_p = D3dynamicLagrangianString(q_km1[1:len(q_k)-1], q_km1[2:len(q_k)],
                                                      q_k[1:len(q_k)-1], q_k[2:len(q_k)],
                                                      C, M, step, time_step)
        momenta_p = D4dynamicLagrangianString(q_km1[:len(q_k)-1], q_km1[1:len(q_k)],
                                                      q_k[:len(q_k)-1], q_k[1:len(q_k)],
                                                      C, M, step, time_step)

        del_eq = jnp.zeros((nsteps,3))
        del_eq = del_eq.at[1:jnp.shape(del_eq)[0]-1, :].add(stress_m + stress_p)
        del_eq = del_eq.at[1:jnp.shape(del_eq)[0], :].add(momenta_m + momenta_p)
        
        return del_eq.flatten()
    
    @jit 
    def jacobianAssembleSystem(q_kp1, q_k, q_km1, string, time_step):
        # jacobian of the discrete Euler Lagrange equations
        return jacfwd(GeometricallyExactString.assembleSystem, 0)(q_kp1, q_k, q_km1, string, time_step)

    @jit
    def assembleSystem1(x_kp1, x_k, v_0, string, time_step):
        # assemble the discrete Euler-Lagrange field equations
        
        step = string.step
        C = string.C
        M = string.M
        nsteps = jnp.shape(x_kp1)[0]//3
        
        q_kp1 = jnp.reshape(x_kp1, (nsteps, 3))
        q_k = jnp.reshape(x_k, (nsteps, 3))
        # vectorize Lagrangian for multiple nodes
        D1dynamicLagrangianString = vmap(GeometricallyExactString.D1dynamicLagrangian, in_axes=[0, 0, 0, 0, None, None, None, None])
        D2dynamicLagrangianString = vmap(GeometricallyExactString.D2dynamicLagrangian, in_axes=[0, 0, 0, 0, None, None, None, None])
        
        # calculate the stresses and momenta
        stress_m = D1dynamicLagrangianString(q_k[0:len(q_k)-1], q_k[1:len(q_k)],
                                                      q_kp1[:len(q_k)-1], q_kp1[1:len(q_k)],
                                                      C, M, step, time_step)
        momenta_m = D2dynamicLagrangianString(q_k[:len(q_k)-1], q_k[1:len(q_k)],
                                                      q_kp1[:len(q_k)-1], q_kp1[1:len(q_k)],
                                                      C, M, step, time_step)
        
        # first time step, derive momenta and stresses from continuous legendre transformation
        p = M * v_0
    
        del_eq = jnp.zeros((nsteps,3))
        del_eq = del_eq.at[1:jnp.shape(del_eq)[0], :].add(stress_m)
        del_eq = del_eq.at[1:jnp.shape(del_eq)[0], :].add(momenta_m)
        
        return (p + del_eq).flatten()
    
    @jit 
    def jacobianAssembleSystem1(q_kp1, q_k, v_0, string, time_step):
        # jacobian of the discrete Euler Lagrange equations
        return jacfwd(GeometricallyExactString.assembleSystem, 0)(q_kp1, q_k, v_0, string, time_step)   
     
if __name__ == "__main__":
    
    
    # create string
    string = GeometricallyExactString.initString(L=3, nsteps=25, rho=4.5, E=1e2, A=1., initialConditions=intiialConditionsString)
    
    # simulation parameters
    T = 5. # simulation time
    nt = 4500 # number of time steps
    dt = T/nt # time step width
    
    # create solution vector
    x = jnp.zeros((3*(string.nsteps+1), nt))
    # set initial position
    x = x.at[:, 0].set(string.q_0.flatten())
    
    # first time step
    sol_dyn = optimize.root(GeometricallyExactString.assembleSystem1,
                        x[:,0],
                        args=(x[:,0], string.v_0, string, dt),
                        method ='hybr',
                        jac=GeometricallyExactString.jacobianAssembleSystem1,
                        options={'xtol': 1e-12})
    x = x.at[:,1].set(sol_dyn.x)
   
    # solve all other time steps
    for n_t in tqdm(range(1, nt+1)):
        sol_dyn = optimize.root(GeometricallyExactString.assembleSystem,
                            x[:,n_t],
                            args=(x[:,n_t], x[:,n_t-1], string, dt),
                            method ='hybr',
                            jac=GeometricallyExactString.jacobianAssembleSystem,
                            options={'xtol': 1e-12})
        x = x.at[:,n_t+1].set(sol_dyn.x)

    # time vector
    t = jnp.linspace(0, T, nt)
    plt.plot(t, x[len(x)-1,:])
    

    x_an = jnp.reshape(x, (string.nsteps+1, 3, nt))
    # create animation
    fig = plt.figure()
    ax = plt.axes(xlim=(1.1*jnp.amin(x_an[:,1,:]), 1.1*jnp.amax(x_an[:,1,:])), ylim=(1.1*jnp.amin(x_an[:,2,:]), 1.1*jnp.amax(x_an[:,2,:])))
    ax.set_aspect('equal')
    line, = ax.plot([], [], lw=3)
    plt.grid(color='0.9')
    
    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x_ani = x_an[:,1,5*i]
        y_ani = x_an[:,2,5*i]
        line.set_data(x_ani, y_ani)
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nt//5, interval=(nt/5)/T, blit=False)
    anim.save('string.gif', writer='imagemagick')