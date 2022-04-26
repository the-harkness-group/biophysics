#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from functools import partial

def twositebinding():

    K1 = 1000
    K2 = 50000
    alpha = 1
    PT = 50e-6
    LT = np.linspace(1e-6,1e-3,25)
    
    delta_F = 1
    delta_B1 = 2
    delta_B2 = 2
    
    PP = np.empty_like(LT,dtype=float)
    PLP = np.empty_like(LT,dtype=float)
    PPL = np.empty_like(LT,dtype=float)
    PLPL = np.empty_like(LT,dtype=float)
    L = np.empty_like(LT,dtype=float)
    
    for i in range(len(LT)):
        #if i == 0:
        #    p =[PT,1e-5,1e-5,1e-5,LT[i]]
        #elif i > 0:
        #    p = [PP[i-1],PPL[i-1],PLP[i-1],PLPL[i-1],L[i-1]]
        #q = [PT,LT[i],K1,K2,alpha]
        #ffs_partial = partial(ffs_complex,q)
        # Solutions are ordered according to how the initial guess vector is arranged
        #PP[i], PPL[i], PLP[i], PLPL[i], L[i] = opt.root(ffs_partial,p,method='lm').x
        
        roots = np.roots([K1*K2, (2.*PT*K1*K2 - LT[i]*K1*K2 + 2.*K1), (2.*PT*K1 - 2.*LT[i]*K1 + 1.), -LT[i]])
        for root in roots:
            if np.isreal(root) and np.real(root) > 0:
                L[i] = np.real(root)

    Q = 1. + 2.*K1*L + K1*K2*np.square(L)
    XF = 1./Q
    XB1 = 2.*K1*L/Q
    XB2 = K1*K2*np.square(L)/Q
        
    delta_obs = delta_F*(XF + (1./2.)*XB1) + (1./2.)*delta_B1*XB1 + delta_B2*XB2
    
    fig, ax = plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(LT*1e6,delta_obs,'o')
    ax[0].set_xlabel('$L_{T}$ $\mu$M')
    ax[0].set_ylabel('$\delta_{obs.}$')
    ax[1].plot(LT*1e6,XF,'o',label='XF')
    ax[1].plot(LT*1e6,XB1,'o',label='XB1')
    ax[1].plot(LT*1e6,XB2,'o',label='XB2')
    ax[1].plot(LT*1e6,XF+XB1+XB2,'o',label='Sum')
    ax[1].set_xlabel('$L_{T}$ $\mu$M')
    ax[1].set_ylabel('Population')
    ax[1].legend()
    
    plt.show()
    
def ffs_complex(q,p):
    
    # Unpack variables and constants
    PP, PPL, PLP, PLPL, L = p # Variables
    PT, LT, K1, K2, alpha = q # Constants

    # Equations have to be set up as equal to 0
    eq1 = -PT + 2.*(PP + PLP + PPL + PLPL)
    eq2 = -LT + L + PLP + PPL + 2*PLPL
    eq3 = -K1 + PLP/(PP*L)
    eq4 = -K1 + PPL/(PP*L)
    eq5 = -K2 + PLPL/(PPL*L)
    
    return [eq1, eq2, eq3, eq4, eq5]

twositebinding()








