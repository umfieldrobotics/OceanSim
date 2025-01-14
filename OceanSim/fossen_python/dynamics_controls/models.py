#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vehicle models.           

Author:     Thor I. Fossen
"""

import numpy as np

def clarke83(U,L,B,T,Cb,R66,xg,T_surge):
    """ 
    [M,N] = clarke83(U,L,B,T,Cb,R66,xg,T_surge) computes the system matrices 
    of a linear maneuvering model based on Clarke et al. (1983). The  
    hydrodynamic derivatives are based on multiple  linear regression from two 
    sets of model tests. The first data set (Yv, Yr, Nv, Nr) is obtained from 
    rotating arm model experiments, while the second data set 
    (Yvdot, Yrdot, Nvdot, Nrdot, Yv, Yr, Nv, Nr) was obtained from a PMM  model.
    
    Outputs: 3x3 model matrices M and N in surge, sway and yaw

       M nu_dot + N(U) nu = tau,     where N(U) = C(U) + D
 
    corresponding to the linear maneuvering model
 
    (m - Xudot) u_dot - Xu u                            = (1-t) T
    (m - Yvdot) v_dot + (m - Yrdot)  r_dot - Yv v - Yr r = Yd delta
    (m - Yvdot) v_dot + (Iz - Nrdot) r_dot - Nv v - Nr r = Nd delta

    Note that the coefficients Yv, Yr, Nv and Nr in the N(U) matrix includes 
    linear damping D and the linearized Coriolis and centripetal matrix C(U).
    
    Inputs:

    U:   speed (m/s)
    L:   length (m)
    B:   beam (m)
    T:   draft (m)
    Cb:  block coefficient (-), Cb = V / (L*B*T) where V is the displaced volume
    R66: radius of gyration in yaw (smaller vessels R66 ≈ 0.25L, tankers R66 ≈ 0.27L)
    xg:  x-coordinate of the CG (m)
    T_surge: time constant in surge (s)

    Reference:  CLARKE, D., GEDLING, P. and HINE. G. (1983). The application of 
    manoeuvring criteria in hull design using linear thory. Trans.  R. lnsm nav. 
    Archit.  125, 45-68. 
    """

    # Rigid-body parameters
    rho = 1025                      # density of water
    V = Cb * L * B * T              # volume displacment
    m = rho * V                     # mass
    Iz = m * R66**2 + m * xg**2     # moment of inerta about the CO
    
    # Rigid-body inertia matrix
    MRB = np.array([
        [ m, 0 ,0], 
        [ 0, m ,m*xg], 
        [ 0, m*xg ,Iz] ],float)

    # Nondimenisonal hydrodynamic derivatives in surge
    Xudot = -0.1 * m
    U += 0.001                 # avoid the singularity for U = 0
    Xu = -( ( m-Xudot )/T_surge ) / ( 0.5 * rho * L**2 * U )
    Xudot = Xudot / ( 0.5 * rho * L**3 )

    # Nondimenisonal hydrodynamic derivatives in sway and yaw
    # from Clarke et al. (1983)
    S = np.pi * (T/L)**2                 # scale factor

    Yvdot = -S * ( 1 + 0.16 * Cb * B/T - 5.1 * (B/L)**2 )
    Yrdot = -S * ( 0.67 * B/L - 0.0033 * (B/T)**2 )
    Nvdot = -S * ( 1.1 * B/L - 0.041 * (B/T) )
    Nrdot = -S * ( 1/12 + 0.017 * Cb * (B/T) - 0.33 * (B/L) )
    Yv = -S * ( 1 + 0.4 * Cb * (B/T) )
    Yr = -S * ( -1/2 + 2.2 * (B/L) - 0.08 * (B/T) )
    Nv = -S * ( 1/2 + 2.4 * (T/L) )
    Nr = -S * ( 1/4 + 0.039 * (B/T) - 0.56 * (B/L) )

    # Nondimenisonal hydrodynamic matrices 
    MA_prime = np.array([
        [ -Xudot, 0 ,0], 
        [ 0, -Yvdot, -Yrdot], 
        [ 0, -Nvdot, -Nrdot] ],float)

    N_prime  = np.array([
        [ -Xu, 0 ,0], 
        [ 0, -Yv, -Yr], 
        [ 0, -Nv, -Nr] ],float)
 
    # Dimensional model (Fossen 2021, Appendix D)   
    T    = np.diag([1, 1, 1/L])
    Tinv = np.diag([1, 1, L])

    MA = (0.5 * rho * L**3) * Tinv @ Tinv @ \
        np.matmul( T, np.matmul( MA_prime,Tinv ) )
                  
    N =  (0.5 * rho * L**2 * U) * Tinv @ Tinv @ \
        np.matmul( T, np.matmul( N_prime, Tinv ) )
 
    M = MRB + MA            # system inertia matrix
    
    return M, N

 
 


