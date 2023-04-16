import numpy as np
import time as clock
# A propagator for numerical integration of a system of ordinary differential equations
# Propagates systems of equations of the form d/dt C(t) = R*C(t)
# C(t) is a vector of time-dependent concentrations, e.g. of molecules
# R is a relaxation matrix containing rate constants that govern fluxes between states
#
# Here is an example for a reversible folding-unfolding reaction of e.g. a small protein
# F <-> U where forward rate constant is k1 and reverse is km1
#
# d/dt C(t) =       R     *  C(t)       
#
# d/dt  [F] =   [-km1 k1]    [F]
#       [U]     [km1 -k1]    [U]
#
# The propagator calculates fluxes to be used in ODE integration to obtain C(t) for all t
# Requires a function func where the relaxation matrix is defined
# This propagation could also be done with a matrix exponential as in NMR analyses

def propagator(t, C, func, constants): # solve_ivp

    R = func(C, constants) # Make relaxation matrix

    return np.matmul(R,C) # Propagate