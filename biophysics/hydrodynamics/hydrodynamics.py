#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:01:23 2021

@author: robertharkness
"""

import numpy as np
from scipy.constants import k as kB, pi

# Calculate solvent viscosity from 3rd order polynomial fits to SEDNTERP buffer viscosity data as a function of temperature
def viscosity(Temperature, eta_coeffs):
    
    if isinstance(eta_coeffs[0],str) == True:
        eta_coeffs = [float(eta_coeffs[x]) for x in range(len(eta_coeffs))]
    
    eta = eta_coeffs[0]*np.power(Temperature,3) + eta_coeffs[1]*np.square(Temperature) + eta_coeffs[2]*Temperature + eta_coeffs[3] # Third order polynomial viscosity as a function of temperature from SEDNTERP

    return eta

### Calculate diffusion coefficients
def stokes_diffusion(Temperature, eta, Rh):
    
    Dt = (kB*Temperature)/(6*np.pi*eta*Rh)

    return Dt

### Calculate hydrodynamic radius
def stokes_radius(Temperature, eta, D):
    
    Rh = (kB*Temperature)/(6*np.pi*eta*D)

    return Rh

# Calculate diffusion coefficients as a function of molecular size according to defined scaling
def scaled_diffusion(Do, size_multiple, exponent, shape='Spherical', f=None):
    
    if shape is 'Spherical':
        D_scaled = Do*size_multiple**exponent
    if shape is 'Linear':
        D_scaled = Do*(size_multiple**exponent)*f
    
    return D_scaled

def disk_rod_diffusion(i, p1, T):

    eta = 0.0011
    L1 = 6.5e-9
    d1 = L1/p1

    L = i*L1
    p = i*p1
    f0 = 6*pi*eta*L*(3/(16*p**2))**(1/3)
    f_f0 = 1.009 + 0.01395*np.log(p) + 0.07880*np.log(p)**2 + 0.00604*np.log(p)**3
    f = f0*f_f0
    D = kB*T/f

    return D

# For self-assembly of monomers
def monomer_diffusion_dictionary(P_dict, shape='Spherical', p=None):
    
    D_dict = P_dict.copy()
    D_dict['size factor'] = np.array([int(k[1:]) for k in P_dict.keys()]) # For Dz calculation
    if shape is 'Spherical':
        D_dict['exponent'] = np.full(len(P_dict.keys()),-0.333) # Spherical scaling factor
        D_dict['f'] = None
    if shape is 'Linear':
        D_dict['exponent'] = np.full(len(P_dict.keys()),-1.0) # Linear scaling factor
        v_correct = np.array([0.632 + (1.165/p[i]) + (0.1/p[i]**2) for i in np.arange(len(p))])
        D_dict['f'] = np.array([2*np.log(p[i]) + v_correct[i] for i in np.arange(len(p))])/(2*np.log(p[0]) + v_correct[0])
    D_dict['Dz'] = np.zeros(1)
    
    return D_dict

# For self-assembly of trimers
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

def calculate_Dz_isothermal(D_dict, C_dict, Do, shape='Spherical'):

    Dz_num = 0 # Get Dz
    Dz_den = 0
    for y, k in enumerate(C_dict.keys()): # C_dict stays same length
        D_dict[k] = scaled_diffusion(Do, D_dict['size factor'][y], D_dict['exponent'][y],
        shape=shape, f=D_dict['f'][y])
        Dz_num += (D_dict['size factor'][y]**2)*C_dict[k]*D_dict[k] # Diffusion
        Dz_den += (D_dict['size factor'][y]**2)*C_dict[k]

    D_dict['Dz'] = Dz_num/Dz_den
    
    return D_dict

def pop_weighted_avg_diff(D_dict, P_dict, Do, shape='Spherical'): # for methods like NMR

    Davg = 0
    for y, k in enumerate(P_dict.keys()): # C_dict stays same length
        D_dict[k] = scaled_diffusion(Do, D_dict['size factor'][y], D_dict['exponent'][y],
        shape=shape, f=D_dict['f'][y])
        Davg += P_dict[k]*D_dict[k]

    D_dict['Dz'] = Davg

    return D_dict

### Calculate scattering vector using detector angle and wavelength
def scattering_vector():
    
    ### Define constants and instrument parameters for Wyatt DynaPRO DLS plate reader
    n = 1.3347
    wavelength = 824e-9
    theta = (150)*(pi/180)
    
    q = (4*pi*n/wavelength)*np.sin(theta/2)
    
    return q

### Simulate DLS autocorrelation function
def autocorrelation(t, D, B, beta, mu2):
    
    q = scattering_vector()
    g2 = B + beta*np.exp(-2.*D*q**2*t)*((1 + (mu2/2.)*t**2)**2)
    
    return g2
    