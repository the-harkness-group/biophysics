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

# Oligomerization through A and B pathway, A pathway stops at hexamer
# B pathway can be infinitely long
def two_pathways(fit_params, fit_constants, temperature, concentration, P_dict, C_dict):

    # Get equilibrium constants
    K1 = bindingmodels.equilibriumconstants(fit_params['K1o'].value,fit_params['dH1o'].value,
            fit_params['dCp1'].value,fit_constants['To'],temperature)
    K2 = bindingmodels.equilibriumconstants(fit_params['K2o'].value,fit_params['dH2o'].value,
            fit_params['dCp2'].value,fit_constants['To'],temperature)

    X_guess = 0.001*concentration*K2/3 # Initial guess for dimensionless trimer concentration to use in solver
    ################## Solve dimensionless trimer concentration ##################
    constants = [(concentration*K2/3),K1/K2] # XT, alpha1 = K1/K2
    equations_partial = partial(equations,constants)
    sol = opt.root(equations_partial,X_guess,method='lm')
    c = sol.x[0]/K2 # Trimer concentration in molar

    # Generate populations and concentrations from solver solutions
    for x in range(1,int(fit_constants['N'])+1):

        if x == 1: # Trimer

            C_dict[f"{3*x}A"] = c
            P_dict[f"{3*x}A"] = bindingmodels.populations(c, 3*x, concentration)

        if x == 2: # Hexamers

            # Hexamer A
            c = K1*(C_dict['3A']**2)
            C_dict[f"{3*x}A"] = c # Hexamer A dictated by trimer and K1
            P_dict[f"{3*x}A"] = bindingmodels.populations(c, 3*x, concentration)

            # Hexamer B
            c = K2*(C_dict['3A']**2)
            C_dict[f"{3*x}B"] = c # Hexamer 2 dictated by trimer and K2
            P_dict[f"{3*x}B"] = bindingmodels.populations(c, 3*x, concentration)

        if x == 3: # 9B

            c = K2*C_dict[f"{3*(x-1)}B"]*C_dict['3A']
            C_dict[f"{3*x}B"] = c # 9-mer dictated by Hexamer 2 and K2
            P_dict[f"{3*x}B"] = bindingmodels.populations(c, 3*x, concentration)

        if x > 3: # >9B

            c = K2*C_dict[f"{3*(x-1)}B"]*C_dict['3A']
            C_dict[f"{3*x}B"] = c # >9-mer dictated by 9-mer and K2
            P_dict[f"{3*x}B"] = bindingmodels.populations(c, 3*x, concentration)

    return C_dict, P_dict

# Oligomerization through A and B pathway, A pathway stops at hexamer
# B pathway can be infinitely long, polynomial root solver
def two_pathways_roots(fit_params, fit_constants, temperature, concentration, P_dict, C_dict):

    # Get equilibrium constants
    K1 = bindingmodels.equilibriumconstants(fit_params['K1o'].value,fit_params['dH1o'].value,
            fit_params['dCp1'].value,fit_constants['To'],temperature)
    K2 = bindingmodels.equilibriumconstants(fit_params['K2o'].value,fit_params['dH2o'].value,
            fit_params['dCp2'].value,fit_constants['To'],temperature)

    ################## Solve dimensionless trimer concentration ##################
    XT = concentration*K2/3
    alpha = K1/K2
    p = [2*alpha,(-4*alpha),((2*alpha)-XT),((2*XT)+1),-XT] # Solve roots of polynomial
    roots = np.roots(p)
    for root in roots:
        if (np.isreal(root)) & (0 <= np.real(root) <= XT): # Get X3 as real positive root
            X3 = np.real(root)
    c = X3/K2 # Trimer concentration in molar

    # Generate populations and concentrations from solver solutions
    for x in range(1,int(fit_constants['N'])+1):

        if x == 1: # Trimer

            C_dict[f"{3*x}A"] = c
            P_dict[f"{3*x}A"] = bindingmodels.populations(c, 3*x, concentration)

        if x == 2: # Hexamers

            # Hexamer A
            c = K1*(C_dict['3A']**2)
            C_dict[f"{3*x}A"] = c # Hexamer A dictated by trimer and K1
            P_dict[f"{3*x}A"] = bindingmodels.populations(c, 3*x, concentration)

            # Hexamer B
            c = K2*(C_dict['3A']**2)
            C_dict[f"{3*x}B"] = c # Hexamer 2 dictated by trimer and K2
            P_dict[f"{3*x}B"] = bindingmodels.populations(c, 3*x, concentration)

        if x == 3: # 9B

            c = K2*C_dict[f"{3*(x-1)}B"]*C_dict['3A']
            C_dict[f"{3*x}B"] = c # 9-mer dictated by Hexamer 2 and K2
            P_dict[f"{3*x}B"] = bindingmodels.populations(c, 3*x, concentration)

        if x > 3: # >9B

            c = K2*C_dict[f"{3*(x-1)}B"]*C_dict['3A']
            C_dict[f"{3*x}B"] = c # >9-mer dictated by 9-mer and K2
            P_dict[f"{3*x}B"] = bindingmodels.populations(c, 3*x, concentration)

    return C_dict, P_dict

### Two pathways, no cooperativity, dimensionless trimer concentration solver
def equations(constants,X):

    XT, a1 = constants # Unpack constants

    eq1 = -XT + 2.*a1*np.square(X) + (X/np.square(X-1))

    return eq1
