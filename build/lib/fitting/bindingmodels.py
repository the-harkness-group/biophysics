#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:37:13 2021

@author: robertharkness
"""

import numpy as np
import copy

# P + L <-> PL
def onesite(LT, PT, K):
    
    a = K
    b = K*PT - K*LT + 1
    c = -LT
    
    L = (-b + np.sqrt(b**2 - 4.*a*c))/(2.*a)
    PL = LT - L
    PB = PL/PT
    
    return PB

# Calculate K according to Ko (ref temperature), dHo, dCp
def equilibriumconstants(Ko, dHo, dCp, To, Temperatures):
    
    R = constants()
    
    K = Ko*np.exp((dHo/R)*((1/To) - (1/Temperatures)) + (dCp/R)*(np.log(Temperatures/To) + (To/Temperatures) - 1))
    
    return K

# Calculate concentrations from populations
def concentrations(P, molecularity, CT):
    
    return P*CT/molecularity

# Calculate concentrations from populations
def populations(c, molecularity, CT):
    
    return molecularity*c/CT

# Define physical constants, e.g. gas constant
def constants():
    
    R = 8.3145e-3 # kJ/mol K
    
    return R

# Make dictionaries needed for the simulations and fitting
def make_dictionaries(state_list):
    
    P_dict = {state:np.zeros(1) for state in state_list}
    
    C_dict = copy.deepcopy(P_dict)
    
    return P_dict, C_dict