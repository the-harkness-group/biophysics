import numpy as np

# Numerical integration of a system of ordinary differential equations
# Solves systems of equations of the form d/dt C(t) = R*C(t)
# C(t) is a vector of time-dependent concentrations, e.g. of molecules
# R is a relaxation matrix containing rate constants that govern fluxes between states
# Here is an example for a folding-unfolding reaction of e.g. a small protein
# F <-> U where forward rate constant is k1 and reverse is km1
#
# d/dt C(t) =       R     *  C(t)       
#
# d/dt  [F] =   [-km1 k1]    [F]
#       [U]     [km1 -k1]    [U]

def rate_eqs(C0, t, params):
    
    R = params # Unpack relaxation matrix

    return np.matmul(R,C0)