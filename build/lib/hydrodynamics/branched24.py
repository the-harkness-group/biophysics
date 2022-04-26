#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 07:39:37 2021

@author: robertharkness
"""

import numpy as np
import bindingmodels
import hydrodynamics
from scipy import optimize as opt
from functools import partial

# Data simulation
def branchedto24(fit_params, P_dict, D_dict, concentration, temperature, To, Rh, eta_coeffs):
    
    K1 = np.abs(fit_params['alpha'].value)*2.*bindingmodels.equilibriumconstants(
    fit_params['K1o'].value,fit_params['dH1o'].value, fit_params['dCp1'].value,
    To, temperature) # Start of Path B, 2 PDZ bound
    K2 = np.abs(fit_params['beta'].value)*4.*bindingmodels.equilibriumconstants(
    fit_params['K1o'].value,fit_params['dH1o'].value,fit_params['dCp1'].value,
    To, temperature) # 4 PDZ bound
    K3 = np.abs(fit_params['gamma'].value)*6.*bindingmodels.equilibriumconstants(
    fit_params['K1o'].value,fit_params['dH1o'].value,fit_params['dCp1'].value,
    To, temperature) # 6 PDZ bound
    K4 = bindingmodels.equilibriumconstants(fit_params['K4o'].value,
    fit_params['dH4o'].value,fit_params['dCp4'].value,To, temperature) # M3 <-> M6A
    
    eta = hydrodynamics.viscosity(temperature, eta_coeffs) # Get diffusion coeffs as a fxn of temperature
    Do = hydrodynamics.stokes_diffusion(temperature, eta, Rh)
    D_dict['D3'] = hydrodynamics.scaled_diffusion(Do, 1, -0.333)        
    D_dict['D6A'] = hydrodynamics.scaled_diffusion(Do, 2, -0.227)
    D_dict['D6B'] = hydrodynamics.scaled_diffusion(Do, 2, -0.227)
    D_dict['D9A'] = hydrodynamics.scaled_diffusion(Do, 3, -0.333)
    D_dict['D9B'] = hydrodynamics.scaled_diffusion(Do, 3, -0.333)
    D_dict['D12A'] = hydrodynamics.scaled_diffusion(Do, 4, -0.333)
    D_dict['D12B'] = hydrodynamics.scaled_diffusion(Do, 4, -0.333)
    D_dict['D12C'] = hydrodynamics.scaled_diffusion(Do, 4, -0.333)
    D_dict['D12D'] = hydrodynamics.scaled_diffusion(Do, 4, -0.333)
    D_dict['D15A'] = hydrodynamics.scaled_diffusion(Do, 5, -0.333)
    D_dict['D15B'] = hydrodynamics.scaled_diffusion(Do, 5, -0.333)
    D_dict['D18A'] = hydrodynamics.scaled_diffusion(Do, 6, -0.333)
    D_dict['D18B'] = hydrodynamics.scaled_diffusion(Do, 6, -0.333)
    D_dict['D21'] = hydrodynamics.scaled_diffusion(Do, 7, -0.333)
    D_dict['D24'] = hydrodynamics.scaled_diffusion(Do, 8, -0.333)
        
    Dz_num = 0
    Dz_den = 0
            
    p =[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025] # Initial population guess
    q = [concentration, K1, K2, K3, K4] # Variables
    equations_partial = partial(equations,q)
    P3, P6A, P6B, P9A, P9B, P12A, P12B, P12C, P12D, P15A, P15B, P18A, P18B, P21, P24  = opt.root(equations_partial,p,method='lm').x
    P_dict['P3'] = P3
    P_dict['P6A'] = P6A
    P_dict['P6B'] = P6B
    P_dict['P9A'] = P9A
    P_dict['P9B'] = P9B
    P_dict['P12A'] = P12A
    P_dict['P12B'] = P12B
    P_dict['P12C'] = P12C
    P_dict['P12D'] = P12D
    P_dict['P15A'] = P15A
    P_dict['P15B'] = P15B
    P_dict['P18A'] = P18A
    P_dict['P18B'] = P18B
    P_dict['P21'] = P21
    P_dict['P24'] = P24
    P_dict['Psum'] = P3 + P6A + P6B + P9A + P9B + P12A + P12B + P12C + P12D + P15A + P15B + P18A + P18B + P21 + P24
    P_dict['FF'] = P3 + (4/6)*P6B + (5/9)*P9A + (3/9)*P9B + (6/12)*P12A + (4/12)*P12C + (4/12)*P12D + (5/15)*P15A + (3/15)*P15B + (4/18)*P18B + (3/21)*P21 # Every possible PDZ1:2 formed
    P_dict['FB'] = (6/6)*P6A + (2/6)*P6B + (4/9)*P9A + (6/9)*P9B + (6/12)*P12A + P12B + (8/12)*P12C + (8/12)*P12D + (10/15)*P15A + (12/15)*P15B + P18A + (14/18)*P18B + (18/21)*P21 + P24 # Every possible PDZ1:2 formed 
    
    M3, M6A, M6B, M9A, M9B, M12A, M12B, M12C, M12D, M15A, M15B, M18A, M18B, M21, M24 = concentrations(P3, P6A, P6B, P9A, P9B, P12A, P12B, P12C, P12D, P15A, P15B, P18A, P18B, P21, P24, concentration)
            
    Dz_num += (1**2)*M3*D_dict['D3']
    Dz_num += (2**2)*M6A*D_dict['D6A']
    Dz_num += (2**2)*M6B*D_dict['D6B']
    Dz_num += (3**2)*M9A*D_dict['D9A']
    Dz_num += (3**2)*M9B*D_dict['D9B']
    Dz_num += (4**2)*M12A*D_dict['D12A']
    Dz_num += (4**2)*M12B*D_dict['D12B']
    Dz_num += (4**2)*M12C*D_dict['D12C']
    Dz_num += (4**2)*M12D*D_dict['D12D']
    Dz_num += (5**2)*M15A*D_dict['D15A']
    Dz_num += (5**2)*M15B*D_dict['D15B']
    Dz_num += (6**2)*M18A*D_dict['D18A']
    Dz_num += (6**2)*M18B*D_dict['D18B']
    Dz_num += (7**2)*M21*D_dict['D21']
    Dz_num += (8**2)*M24*D_dict['D24']
            
    Dz_den += (1**2)*M3
    Dz_den += (2**2)*M6A
    Dz_den += (2**2)*M6B
    Dz_den += (3**2)*M9A
    Dz_den += (3**2)*M9B
    Dz_den += (4**2)*M12A
    Dz_den += (4**2)*M12B
    Dz_den += (4**2)*M12C
    Dz_den += (4**2)*M12D
    Dz_den += (5**2)*M15A
    Dz_den += (5**2)*M15B
    Dz_den += (6**2)*M18A
    Dz_den += (6**2)*M18B
    Dz_den += (7**2)*M21
    Dz_den += (8**2)*M24
            
    D_dict['Dz'] = Dz_num/Dz_den
            
    return P_dict, D_dict

# Make dictionaries needed for the simulations and fitting
def makedictionaries():
    
    P_dict = {'P3':np.zeros(1),'P6A':np.zeros(1),'P6B':np.zeros(1),'P9A':np.zeros(1),'P9B':np.zeros(1),
              'P12A':np.zeros(1),'P12B':np.zeros(1), 'P12C':np.zeros(1), 'P12D':np.zeros(1),
              'P15A':np.zeros(1), 'P15B':np.zeros(1), 'P18A':np.zeros(1), 'P18B':np.zeros(1),
              'P21':np.zeros(1), 'P24':np.zeros(1),'FF':np.zeros(1), 'FB':np.zeros(1), 'Psum':np.zeros(1)}
    
    D_dict = {'D3':np.zeros(1),'D6A':np.zeros(1),'D6B':np.zeros(1),
              'D9A':np.zeros(1),'D9B':np.zeros(1),'D12A':np.zeros(1),
              'D12B':np.zeros(1),'D12C':np.zeros(1),'D12D':np.zeros(1),
              'D15A':np.zeros(1),'D15B':np.zeros(1),'D18A':np.zeros(1),
              'D18B':np.zeros(1),'D21':np.zeros(1),'D24':np.zeros(1),
              'Dz':np.zeros(1)}
    
    return P_dict, D_dict

# Calculate concentrations from populations
def concentrations(P3, P6A, P6B, P9A, P9B, P12A, P12B, P12C, P12D, P15A, P15B, P18A, P18B, P21, P24, CT):
    
    M3 = P3*CT/3
    M6A = P6A*CT/6
    M6B = P6B*CT/6
    M9A = P9A*CT/9
    M9B = P9B*CT/9
    M12A = P12A*CT/12
    M12B = P12B*CT/12
    M12C = P12C*CT/12
    M12D = P12D*CT/12
    M15A = P15A*CT/15
    M15B = P15B*CT/15
    M18A = P18A*CT/18
    M18B = P18B*CT/18
    M21 = P21*CT/21
    M24 = P24*CT/24
    
    return M3, M6A, M6B, M9A, M9B, M12A, M12B, M12C, M12D, M15A, M15B, M18A, M18B, M21, M24

# System of equations
def equations(q, p):
    
    P3, P6A, P6B, P9A, P9B, P12A, P12B, P12C, P12D, P15A, P15B, P18A, P18B, P21, P24 = p # Variables
    CT, K1, K2, K3, K4 = q # Constants
    
    M3, M6A, M6B, M9A, M9B, M12A, M12B, M12C, M12D, M15A, M15B, M18A, M18B, M21, M24 = concentrations(P3, P6A, P6B, P9A, P9B, P12A, P12B, P12C, P12D, P15A, P15B, P18A, P18B, P21, P24, CT)
    
    eq1 = -1 + P3 + P6A + P6B + P9A + P9B + + P12A + P12B + P12C + P12D + P15A + P15B + P18A + P18B + P21 + P24
    eq2 = K4*M3**2 - M6A # Using K from DLS as is, already built in statistical weighting
    eq3 = (9/1)*K1*M3**2 - M6B
    eq4 = (12/2)*K1*M6B*M3 - M9A
    eq5 = (6/3)*K2*M6B*M3 - M9B
    eq6 = (15/7)*K1*M9A*M3 - M12A
    eq7 = (3/4)*K3*M9B*M3 - M12B
    eq8 = (9/1)*K1*M9B*M3 - M12C
    eq9 = (3/4)*K2*M9A*M3 - M12D
    eq10 = (12/1)*K1*M12D*M3 - M15A
    eq11 = (6/1)*K2*M12C*M3 - M15B
    eq12 = (3/6)*K3*M15B*M3 - M18A
    eq13 = (6/2)*K2*M15A*M3 - M18B
    eq14 = (6/2)*K2*M18B*M3 - M21
    eq15 = (3/8)*K3*M21*M3 - M24
    
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15]