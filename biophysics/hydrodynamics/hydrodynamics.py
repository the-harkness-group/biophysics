#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:01:23 2021

@author: robertharkness
"""

import numpy as np

# Calculate solvent viscosity from 3rd order polynomial fits to SEDNTERP buffer viscosity data as a function of temperature
def viscosity(Temperature, eta_coeffs):
    
    if isinstance(eta_coeffs[0],str) == True:
        eta_coeffs = [float(eta_coeffs[x]) for x in range(len(eta_coeffs))]
    
    eta = eta_coeffs[0]*np.power(Temperature,3) + eta_coeffs[1]*np.square(Temperature) + eta_coeffs[2]*Temperature + eta_coeffs[3] # Third order polynomial viscosity as a function of temperature from SEDNTERP

    return eta

### Calculate diffusion coefficients
def stokes_diffusion(Temperature, eta, Rh):
    
    kB = 1.38065e-23
    Dt = (kB*Temperature)/(6*np.pi*eta*Rh)

    return Dt

### Calculate hydrodynamic radius
def stokes_radius(Temperature, eta, D):
    
    kB = 1.38065e-23
    Rh = (kB*Temperature)/(6*np.pi*eta*D)

    return Rh

# Calculate diffusion coefficients as a function of molecular size according to defined scaling
def scaled_diffusion(Do, size_multiple, exponent):
    
    D_scaled = Do*size_multiple**exponent
    
    return D_scaled

def make_diffusion_dictionary(P_dict):
    
    D_dict = P_dict.copy()
    D_dict['size factor'] = np.array([int(k[:-1])/3 for k in P_dict.keys()]) # For Dz calculation
    D_dict['exponent'] = np.full(len(P_dict.keys()),-0.333) # Spherical scaling
    D_dict['exponent'][1] = -0.227 # Hexamers have non-spherical scaling
    D_dict['exponent'][2] = -0.227 # Hexamers have non-spherical scaling

    D_dict['Dz'] = np.zeros(1)
    
    return D_dict

def calculate_Dz(D_dict, C_dict, temperature, eta_coeffs, Rho):
    
    Dz_num = 0 # Get Dz
    Dz_den = 0
    eta = viscosity(temperature, eta_coeffs) # Get diffusion coeffs as a fxn of temperature
    Do = stokes_diffusion(temperature, eta, Rho)
    for y, k in enumerate(C_dict.keys()): # C_dict stays same length
        D_dict[k] = scaled_diffusion(Do, D_dict['size factor'][y], D_dict['exponent'][y])
        Dz_num += (D_dict['size factor'][y]**2)*C_dict[k]*D_dict[k] # Diffusion
        Dz_den += (D_dict['size factor'][y]**2)*C_dict[k]
    D_dict['Dz'] = Dz_num/Dz_den
    
    return D_dict
    