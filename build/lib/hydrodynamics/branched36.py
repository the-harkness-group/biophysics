#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:40:55 2021

@author: robertharkness
"""

import numpy as np
import bindingmodels
import hydrodynamics
from scipy import optimize as opt
from functools import partial

# Data simulation
def branchedto36(fit_params, P_dict, D_dict, C_dict, concentration, temperature, To, Rh, eta_coeffs):
    
    K1 = bindingmodels.equilibriumconstants(fit_params['K1o'].value,
    fit_params['dH1o'].value,fit_params['dCp1'].value,To, temperature) # M3 <-> M6A
    K2 = np.abs(fit_params['alpha'].value)*2.*bindingmodels.equilibriumconstants(
    fit_params['K2o'].value,fit_params['dH2o'].value, fit_params['dCp2'].value,
    To, temperature) # Start of Path B, 2 PDZ bound
    K3 = np.abs(fit_params['beta'].value)*4.*bindingmodels.equilibriumconstants(
    fit_params['K2o'].value,fit_params['dH2o'].value,fit_params['dCp2'].value,
    To, temperature) # 4 PDZ bound
    K4 = np.abs(fit_params['gamma'].value)*6.*bindingmodels.equilibriumconstants(
    fit_params['K2o'].value,fit_params['dH2o'].value,fit_params['dCp2'].value,
    To, temperature) # 6 PDZ bound
    
#    K2 = 2.*bindingmodels.equilibriumconstants(fit_params['K2o'].value,
#    fit_params['dH2o'].value, fit_params['dCp2'].value, To, temperature) # Start of Path B, 2 PDZ bound
#    K3 = 4.*bindingmodels.equilibriumconstants(fit_params['K3o'].value,
#    fit_params['dH3o'].value,fit_params['dCp3'].value, To, temperature) # 4 PDZ bound
#    K4 = 6.*bindingmodels.equilibriumconstants(fit_params['K4o'].value,
#    fit_params['dH4o'].value,fit_params['dCp4'].value, To, temperature) # 6 PDZ bound
    
#    K1 = np.abs(K1)
#    K2 = np.abs(K2)
#    K3 = np.abs(K3)
#    K4 = np.abs(K4)
    
#    K2 = np.abs(K2)
#    K3 = np.abs(K2)
#    K4 = np.abs(K2)
    
    p =[0.01 for x in range(len(C_dict.keys()))] # Initial population guess
    q = [concentration, K1, K2, K3, K4] # Variables
    equations_partial = partial(equations,q)
    sol  = opt.root(equations_partial,p,method='lm').x # Solve populations
    
    Dz_num = 0 # Get Dz
    Dz_den = 0
    eta = hydrodynamics.viscosity(temperature, eta_coeffs) # Get diffusion coeffs as a fxn of temperature
    Do = hydrodynamics.stokes_diffusion(temperature, eta, Rh)
    for y, k in enumerate(C_dict.keys()): # C_dict stays same length
        P_dict[k] = sol[y] # Population
        C_dict[k] = concentrations(P_dict[k], int(k[:-1]), concentration) # Concentration
        D_dict[k] = hydrodynamics.scaled_diffusion(Do, D_dict['size factor'][y], D_dict['exponent'][y])
        Dz_num += (D_dict['size factor'][y]**2)*C_dict[k]*D_dict[k] # Diffusion
        Dz_den += (D_dict['size factor'][y]**2)*C_dict[k]
    P_dict['Psum'] = np.sum(sol)
    D_dict['Dz'] = Dz_num/Dz_den
    P_dict = fractionbound(P_dict)
            
    return P_dict, D_dict, C_dict

# Make dictionaries needed for the simulations and fitting
def makedictionaries():
    
    P_dict = {'3A':np.zeros(1),'6A':np.zeros(1),'6B':np.zeros(1),'9B':np.zeros(1),
    '9C':np.zeros(1),'12B':np.zeros(1),'12C':np.zeros(1),'12D':np.zeros(1),'12E':np.zeros(1),
    '15B':np.zeros(1),'15D':np.zeros(1),'15E':np.zeros(1),'15F':np.zeros(1), 
    '18D':np.zeros(1),'18E':np.zeros(1),'18F':np.zeros(1),'18G':np.zeros(1),
    '21E':np.zeros(1),'21F':np.zeros(1),'21G':np.zeros(1),'24E':np.zeros(1),
    '24F':np.zeros(1),'24G':np.zeros(1),'27F':np.zeros(1),'27G':np.zeros(1),
    '30F':np.zeros(1),'30G':np.zeros(1),'33G':np.zeros(1),'36G':np.zeros(1)}
    
#    P_dict = {'3A':np.zeros(1),'6A':np.zeros(1),'6B':np.zeros(1),'9B':np.zeros(1),
#    '12B':np.zeros(1),'12E':np.zeros(1),
#    '15B':np.zeros(1),'15E':np.zeros(1),'15F':np.zeros(1), 
#    '18E':np.zeros(1),'18F':np.zeros(1),'18G':np.zeros(1),
#    '21E':np.zeros(1),'21F':np.zeros(1),'21G':np.zeros(1),'24E':np.zeros(1),
#    '24F':np.zeros(1),'24G':np.zeros(1),'27F':np.zeros(1),'27G':np.zeros(1),
#    '30F':np.zeros(1),'30G':np.zeros(1),'33G':np.zeros(1),'36G':np.zeros(1)}
    
    C_dict = P_dict.copy()
    D_dict = P_dict.copy()
    D_dict['size factor'] = np.array([int(k[:-1])/3 for k in P_dict.keys()]) # For Dz calculation
    D_dict['exponent'] = np.full(len(P_dict.keys()),-0.333) # Spherical scaling
    D_dict['exponent'][1] = -0.227 # Hexamers have non-spherical scaling
    D_dict['exponent'][2] = -0.227
    D_dict['Dz'] = np.zeros(1)
    
    return P_dict, D_dict, C_dict

# Calculate concentrations from populations
def concentrations(P, molecularity, CT):
    
    return P*CT/molecularity

# System of equations, branched oligomerization to 36-mer
def equations(q, p):
    
    (P3A, P6A, P6B, P9B, P9C, P12B, P12C, P12D, P12E, P15B, P15D, P15E, P15F, 
    P18D, P18E, P18F, P18G, P21E, P21F, P21G, P24E, P24F, P24G, P27F, P27G,
    P30F, P30G, P33G, P36G) = p # Variables
     
#    (P3A, P6A, P6B, P9B, P12B, P12E, P15B, P15E, P15F, 
#    P18E, P18F, P18G, P21E, P21F, P21G, P24E, P24F, P24G, P27F, P27G,
#    P30F, P30G, P33G, P36G) = p # Variables
    CT, K1, K2, K3, K4 = q # Constants
    
    M3A = concentrations(P3A, 3, CT)
    M6A = concentrations(P6A, 6, CT)
    M6B = concentrations(P6B, 6, CT)
    M9B = concentrations(P9B, 9, CT)
    M9C = concentrations(P9C, 9, CT)
    M12B = concentrations(P12B, 12, CT)
    M12C = concentrations(P12C, 12, CT)
    M12D = concentrations(P12D, 12, CT)
    M12E = concentrations(P12E, 12, CT)
    M15B = concentrations(P15B, 15, CT)
    M15D = concentrations(P15D, 15, CT)
    M15E = concentrations(P15E, 15, CT)
    M15F = concentrations(P15F, 15, CT)
    M18D = concentrations(P18D, 18, CT)
    M18E = concentrations(P18E, 18, CT)
    M18F = concentrations(P18F, 18, CT)
    M18G = concentrations(P18G, 18, CT)
    M21E = concentrations(P21E, 21, CT)
    M21F = concentrations(P21F, 21, CT)
    M21G = concentrations(P21G, 21, CT)
    M24E = concentrations(P24E, 24, CT)
    M24F = concentrations(P24F, 24, CT)
    M24G = concentrations(P24G, 24, CT)
    M27F = concentrations(P27F, 27, CT)
    M27G = concentrations(P27G, 27, CT)
    M30F = concentrations(P30F, 30, CT)
    M30G = concentrations(P30G, 30, CT)
    M33G = concentrations(P33G, 33, CT)
    M36G = concentrations(P36G, 36, CT)
    
    eq1 = (-1 + P3A + P6A + P6B + P9B + P9C + P12B + P12C + P12D + P12E + # Total population
           P15B + P15D + P15E + P15F + P18D + P18E + P18F + P18G + P21E + P21F
          + P21G + P24E + P24F + P24G + P27F + P27G + P30F + P30G + P33G + P36G)
    
#    eq1 = (-1 + P3A + P6A + P6B + P9B + P12B + P12E + # Total population
#           P15B + P15E + P15F + P18E + P18F + P18G + P21E + P21F
#          + P21G + P24E + P24F + P24G + P27F + P27G + P30F + P30G + P33G + P36G)
    
    eq2 = (9/1)*K1*M3A**2 - M6A # A arm
    
    eq3 = (9/1)*K2*M3A**2 - M6B # B arm
    eq4 = (12/2)*K2*M6B*M3A - M9B
    eq5 = (6/2)*K2*M9B*M3A - M12B
    #eq5 = (12/2)*K2*M9B*M3A - M12B
    eq6 = (6/2)*M12B*M3A - M15B
    #eq6 = (12/2)*M12B*M3A - M15B
    
    eq7 = (6/3)*K3*M6B*M3A - M9C # C arm
    eq8 = (3/4)*K4*M9C*M3A - M12C
    
    eq9 = (9/1)*K2*M9C*M3A - M12D # D arm
    eq10 = (6/2)*K3*M12D*M3A - M15D
    eq11 = (3/6)*K4*M15D*M3A - M18D
    
    eq12 = (3/4)*K3*M9B*M3A - M12E # E arm
    #eq12 = (6/4)*K3*M9B*M3A - M12E
    eq13 = (12/1)*K2*M12E*M3A - M15E
    eq14 = (6/2)*K3*M15E*M3A - M18E
    eq15 = (6/2)*K3*M18E*M3A - M21E
    eq16 = (3/8)*K4*M21E*M3A - M24E
    
    eq17 = (3/5)*K3*M12B*M3A - M15F # F arm
    #eq17 = (6/5)*K3*M12B*M3A - M15F
    eq18 = (15/1)*K2*M15F*M3A - M18F
    eq19 = (6/2)*K3*M18F*M3A - M21F
    eq20 = (6/2)*K3*M21F*M3A - M24F
    eq21 = (6/2)*K3*M24F*M3A - M27F
    eq22 = (3/10)*K4*M27F*M3A - M30F
    
    eq23 = (3/6)*K3*M15B*M3A - M18G # G arm
    #eq23 = (6/6)*K3*M15B*M3A - M18G
    eq24 = (18/1)*K2*M18G*M3A - M21G
    eq25 = (6/2)*K3*M21G*M3A - M24G
    eq26 = (6/2)*K3*M24G*M3A - M27G
    eq27 = (6/2)*K3*M27G*M3A - M30G
    eq28 = (6/2)*K3*M30G*M3A - M33G
    eq29 = (3/12)*K4*M33G*M3A - M36G
    
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, 
            eq13, eq14, eq15, eq16, eq17, eq18, eq19, eq20, eq21, eq22, eq23,
            eq24, eq25, eq26, eq27, eq28, eq29]
    
#    return [eq1, eq2, eq3, eq4, eq5, eq6, eq12, 
#            eq13, eq14, eq15, eq16, eq17, eq18, eq19, eq20, eq21, eq22, eq23,
#            eq24, eq25, eq26, eq27, eq28, eq29]
    
def fractionbound(P_dict):
    
    # To 24-mer
#    P_dict['FB'] = (P_dict['6A'] + (2/6)*P_dict['6B'] + (4/9)*P_dict['9B'] +
#    (6/12)*P_dict['12B'] + (8/15)*P_dict['15B'] + (6/9)*P_dict['9C'] +
#    P_dict['12C'] + (8/12)*P_dict['12D'] + (12/15)*P_dict['15D'] + P_dict['18D']
#    + (8/12)*P_dict['12E'] + (10/15)*P_dict['15E'] + (14/18)*P_dict['18E'] +
#    (17/21)*P_dict['21E'] + P_dict['24E'] + (10/15)*P_dict['15F'] + (12/18)*P_dict['18F']
#    + (16/21)*P_dict['21F'] + (19/21)*P_dict['24F'] + (12/18)*P_dict['18G'] +
#    (14/21)*P_dict['21G'] + (18/24)*P_dict['24G'])
#    
#    P_dict['FB'] = (P_dict['FB']/(P_dict['3A'] + P_dict['6A'] + P_dict['6B'] +
#          P_dict['9B'] + P_dict['12B'] + P_dict['15B'] + P_dict['9C'] + P_dict['12C']
#    + P_dict['12D'] + P_dict['15D'] + P_dict['18D'] + P_dict['12E'] + P_dict['15E']
#    + P_dict['18E'] + P_dict['21E'] + P_dict['24E'] + P_dict['15F'] + P_dict['18F']
#    + P_dict['21F'] + P_dict['24F'] + P_dict['18G'] + P_dict['21G'] + P_dict['24G']))
    
#    P_dict['FB'] = (P_dict['6A'] + (2/6)*P_dict['6B'] + (4/9)*P_dict['9B'] +
#    (6/12)*P_dict['12B'] + (8/15)*P_dict['15B'] +
#    (8/12)*P_dict['12E'] + (10/15)*P_dict['15E'] + (14/18)*P_dict['18E'] +
#    (17/21)*P_dict['21E'] + P_dict['24E'] + (10/15)*P_dict['15F'] + (12/18)*P_dict['18F']
#    + (16/21)*P_dict['21F'] + (19/21)*P_dict['24F'] + (12/18)*P_dict['18G'] +
#    (14/21)*P_dict['21G'] + (18/24)*P_dict['24G'])
#    
#    P_dict['FB'] = (P_dict['FB']/(P_dict['3A'] + P_dict['6A'] + P_dict['6B'] +
#          P_dict['9B'] + P_dict['12B'] + P_dict['15B'] +
#    P_dict['12E'] + P_dict['15E']
#    + P_dict['18E'] + P_dict['21E'] + P_dict['24E'] + P_dict['15F'] + P_dict['18F']
#    + P_dict['21F'] + P_dict['24F'] + P_dict['18G'] + P_dict['21G'] + P_dict['24G']))
#    
    # To 12-mer
    P_dict['FB'] = (P_dict['6A'] + (2/6)*P_dict['6B'] + (4/9)*P_dict['9B'] +
    (6/12)*P_dict['12B'] + (6/9)*P_dict['9C'] + P_dict['12C'] +
    (8/12)*P_dict['12D'] + (8/12)*P_dict['12E'])
    
#    P_dict['FB'] = (P_dict['6A'] + (1/6)*P_dict['6B'] + (2/9)*P_dict['9B'] +
#    (3/12)*P_dict['12B'] + (3/9)*P_dict['9C'] + (6/12)*P_dict['12C'] +
#    (4/12)*P_dict['12D'] + (4/12)*P_dict['12E'])
    
    P_dict['FB'] = (P_dict['FB']/(P_dict['3A'] + P_dict['6A'] + P_dict['6B'] +
    P_dict['9B'] + P_dict['12B'] + P_dict['9C'] + P_dict['12C']
    + P_dict['12D'] + P_dict['12E']))
    
    # To 12-mer
#    P_dict['FB'] = (P_dict['6A'] + (2/6)*P_dict['6B'] + (4/9)*P_dict['9B'] +
#    (6/12)*P_dict['12B'] + (8/12)*P_dict['12E'])
#    
#    P_dict['FB'] = (P_dict['FB']/(P_dict['3A'] + P_dict['6A'] + P_dict['6B'] +
#          P_dict['9B'] + P_dict['12B'] + P_dict['12E']))
    
    # To 9-mer
#    P_dict['FB'] = (P_dict['6A'] + (2/6)*P_dict['6B'] + (4/9)*P_dict['9B']
#    + (6/9)*P_dict['9C'])
#    
#    P_dict['FB'] = (P_dict['FB']/(P_dict['3A'] + P_dict['6A'] + P_dict['6B'] +
#          P_dict['9B'] + P_dict['9C']))
    
#    P_dict['FB'] = (P_dict['6A'] + (2/6)*P_dict['6B'] + (4/9)*P_dict['9B'])
#    
#    P_dict['FB'] = (P_dict['FB']/(P_dict['3A'] + P_dict['6A'] + P_dict['6B'] +
#          P_dict['9B']))
          
    return P_dict