#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:23:07 2021

@author: harkness
"""

import numpy as np
from biophysics.thermodynamics import bindingmodels
from functools import partial
import scipy.optimize as opt

# Oligomerization through a single polymerization pathway, infinite association of monomers
def one_pathway(fit_params, fit_constants, temperature, concentration, P_dict, C_dict):

    # Get equilibrium constants
    #K1 = bindingmodels.equilibriumconstants(fit_params['K1o'].value,fit_params['dH1o'].value,
    #fit_params['dCp1'].value,fit_constants['To'],temperature)

    K1 = fit_params['K1o']

    X_guess = 0.00001*concentration*K1 # Initial guess for dimensionless monomer concentration
    ################## Solve dimensionless monomer concentration ##################
    constants = concentration*K1 # XT
    equations_partial = partial(equations,constants)
    sol = opt.root(equations_partial,X_guess,method='lm')
    c = sol.x[0]/K1 # Monomer concentration in molar

    # Generate populations and concentrations from solver solutions
    for x in range(1,int(fit_constants['N'])+1):

        if x == 1: # Monomer

            C_dict[f"M{x}"] = c
            P_dict[f"M{x}"] = bindingmodels.populations(c, x, concentration)

        if x >=2: # Everything else

            c = K1*(C_dict[f"M{x-1}"]*C_dict['M1'])
            C_dict[f"M{x}"] = c
            P_dict[f"M{x}"] = bindingmodels.populations(c, x, concentration)

    return C_dict, P_dict

### Two pathways, no cooperativity, dimensionless trimer concentration solver
def equations(constants,X):

    XT = constants # Unpack constants

    eq = -XT + (X/np.square(X-1))

    return eq
