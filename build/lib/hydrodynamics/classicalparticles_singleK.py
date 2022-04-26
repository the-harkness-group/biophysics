#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:42:01 2021

@author: robertharkness
"""

import numpy as np
import bindingmodels
from functools import partial
import scipy.optimize as opt

# Discrete particles: 6, 12, 24, single K value
def classical_3state(fit_params, fit_constants, temperature, concentration, P_dict, C_dict):

    # Get equilibrium constants
    K1 = np.abs(bindingmodels.equilibriumconstants(fit_params['K1o'].value,fit_params['dH1o'].value,
            fit_params['dCp1'].value,fit_constants['To'],temperature))
    
    ################## Solve trimer population ##################
    guess_pops = [0.05,0.05,0.05]
    constants = [concentration, K1]
    equations_partial = partial(eqs_3state,constants)
    sol = opt.root(equations_partial,guess_pops,method='lm')
    
    # Get populations of M6, M12, M24
    C_dict['6A'] = sol.x[0]*concentration/6 # 6-merconcentration in molar
    C_dict['12B'] = sol.x[1]*concentration/12 # 12-mer concentration in molar
    C_dict['24B'] = sol.x[2]*concentration/24 # 24-mer concentration in molar
    P_dict['6A'] = sol.x[0] # Populations are solutions
    P_dict['12B'] = sol.x[1]
    P_dict['24B'] = sol.x[2]
    
    return C_dict, P_dict

# Discrete particles: 3, 6, 12, 24, single K value
def classical_4state(fit_params, fit_constants, temperature, concentration, P_dict, C_dict):

    # Get equilibrium constants
    K1 = np.abs(bindingmodels.equilibriumconstants(fit_params['K1o'].value,fit_params['dH1o'].value,
            fit_params['dCp1'].value,fit_constants['To'],temperature))
    
    ################## Solve trimer population ##################
    guess_pops = [0.05,0.05,0.05,0.05]
    constants = [concentration, K1]
    equations_partial = partial(eqs_4state,constants)
    sol = opt.root(equations_partial,guess_pops,method='lm')
    
    # Get populations of M3, M6, M12, M24
    C_dict['3A'] = sol.x[0]*concentration/3 # Trimer concentration in molar
    C_dict['6A'] = sol.x[1]*concentration/6 # 6-mer concentration in molar
    C_dict['12B'] = sol.x[2]*concentration/12 # 12-mer concentration in molar
    C_dict['24B'] = sol.x[3]*concentration/24 # 24-mer concentration in molar
    P_dict['3A'] = sol.x[0] # Populations are solutions
    P_dict['6A'] = sol.x[1]
    P_dict['12B'] = sol.x[2]
    P_dict['24B'] = sol.x[3]
    
    return C_dict, P_dict

#### One pathway, discrete particles: 6, 12, 24
def eqs_3state(constants,X):

    MT, K1 = constants # Unpack constants
    
    P6, P12, P24 = X
    
    M6 = P6*MT/6
    M12 = P12*MT/12
    M24 = P24*MT/24
    
    eq1 = -1 + P6 + P12 + P24
    eq2 = K1*M6**2 - M12
    eq3 = K1*M12**2 - M24

    return [eq1, eq2, eq3]

#### One pathway, discrete particles: 3, 6, 12, 24
def eqs_4state(constants,X):

    MT, K1 = constants # Unpack constants
    
    P3, P6, P12, P24 = X
    
    M3 = P3*MT/3
    M6 = P6*MT/6
    M12 = P12*MT/12
    M24 = P24*MT/24
    
    eq1 = -1 + P3 + P6 + P12 + P24
    eq2 = K1*M3**2 - M6
    eq3 = K1*M6**2 - M12
    eq4 = K1*M12**2 - M24

    return [eq1, eq2, eq3, eq4]
