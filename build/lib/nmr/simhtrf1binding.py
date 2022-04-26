#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from functools import partial

def main():

    # Experimental concentrations
    LT = np.linspace(0e-6,10000e-6,100)
    PT = np.array(346e-6)

    # Thermo params
    K1 = 8
    K2 = 1/5e-6
    K3 = 1/5e-6
    K4 = 1/5e-6
    
    # Simulate
    solutions = []
    for ligand in LT:
        p = [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]
        q = [ligand,PT,K1,K2,K3,K4]
        eqns_partial = partial(eqns,q)
        # Solutions are ordered according to how the initial guess vector is arranged
        solutions.append(opt.root(eqns_partial,p,method='lm'))

    # Get populations
    Pop_dict = Populations(PT,LT,solutions)
    FB_simple = onesite(LT,PT,K2)

    # Call plotting function with simulation/fit result
    plot_result(LT,PT,Pop_dict,FB_simple)

# Eqns for simulation
def eqns(q,p):
    
    # Unpack variables and constants
    F, U, P3, P3U, P3U2, P3U3 = p # Variables
    LT, PT, K1, K2, K3, K4 = q # Constants

    eqs = []
    eqs.append(-LT + 3*P3 + 3*P3U + 3*P3U2 + 3*P3U3)
    eqs.append(-PT + F + U + P3U + 2*P3U2 + 3*P3U3)
    eqs.append(K1*U - F)
    eqs.append(3*K2*P3*U - P3U)
    eqs.append(K3*P3U*U - P3U2)
    eqs.append((1/3)*K4*P3U2*U - P3U3)

    return eqs

# Calculate populations from the solver solutions
def Populations(PT,LT,solutions):
    
    Pop_dict = {'F':[],'U':[],'P3':[],'P3U':[],'P3U2':[],'P3U3':[]}
    for sol,ligand in zip(solutions,LT):
        Pop_dict['F'].append(np.array(sol.x[0])/PT)
        Pop_dict['U'].append(np.array(sol.x[1])/PT)
        Pop_dict['P3'].append(3*np.array(sol.x[2])/ligand)
        Pop_dict['P3U'].append(np.array(sol.x[3])/PT)
        Pop_dict['P3U2'].append(2*np.array(sol.x[4])/PT)
        Pop_dict['P3U3'].append(3*np.array(sol.x[5])/PT)

    return Pop_dict

# P + L <-> PL
def onesite(LT, PT, K):
    
    a = K
    b = K*PT - K*LT + 1
    c = -LT
    
    L = (-b + np.sqrt(b**2 - 4.*a*c))/(2.*a)
    PL = LT - L
    PB = PL/PT
    
    return PB

# Plot simulations
def plot_result(LT,PT,Pop_dict, FB_simple):

    plt.style.use('figure')
    fig,ax = plt.subplots(1,1)
    LT = LT*1e6
    ax.plot(LT,Pop_dict['F'],label='F')
    ax.plot(LT,Pop_dict['U'],label='U')
    ax.plot(LT,Pop_dict['P3'],label='$P_{3}$')
    ax.plot(LT,Pop_dict['P3U'],label='$P_{3}U$')
    ax.plot(LT,Pop_dict['P3U2'],label='$P_{3}U_{2}$')
    ax.plot(LT,Pop_dict['P3U3'],label='$P_{3}U_{3}$')
    ax.plot(LT,np.array(Pop_dict['P3U'])+np.array(Pop_dict['P3U2'])+np.array(Pop_dict['P3U3']),label='$F_{B}$')
    ax.plot(LT,FB_simple,'--',label='2-state')
    ax.plot(LT,np.array(Pop_dict['F'])+np.array(Pop_dict['U'])+np.array(Pop_dict['P3U'])+np.array(Pop_dict['P3U2'])+np.array(Pop_dict['P3U3']),label='Sum')
    ax.set_xlabel('$L_{T}$ $\mu$M')
    ax.set_ylabel('Fraction')
    ax.set_xlim([0,max(LT)])
    ax.set_xticks([0,max(LT)*0.2,max(LT)*0.4,max(LT)*0.6,max(LT)*0.8,max(LT)])
    ax.legend(loc='upper right',fontsize=12,markerscale=2)
    fig.tight_layout()
    plt.savefig('sims.pdf',format='pdf')

main()