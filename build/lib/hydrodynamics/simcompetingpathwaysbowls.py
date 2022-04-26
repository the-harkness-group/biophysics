#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:12:10 2020

@author: robertharkness
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from functools import partial


def main():
    
    # Protein params
    PT = np.linspace(2e-6,800e-6,100)
    
    # Thermo params
    R = 1.9872e-3
    T = 25 + 273.15
    dG1 = -8
    dG2 = -2
    
    P_dict = populations(PT, R, T, dG1, dG2)
    
    plotting(PT,P_dict)
    
    
# Simulate populations
def populations(PT, R, T, dG1, dG2):
    
    K1 = np.exp(-dG1/(R*T))
    K2 = np.exp(-dG2/(R*T))
    K3 = np.exp(-2*dG2/(R*T))
    K4 = np.exp(-dG2/(R*T))
    K5 = np.exp(-3*dG2/(R*T))
    K6 = np.exp(-2*dG2/(R*T))
    K7 = np.exp(-3*dG2/(R*T))
    
    # Solve for unknown concentrations numerically using fsolve
    solutions = [None]*len(PT) # List of solutions
    p = [PT[0]*0.01,PT[0]*0.01,PT[0]*0.01,PT[0]*0.01,PT[0]*0.01,PT[0]*0.01,PT[0]*0.01,PT[0]*0.01] # initial guesses
    P_dict = {'P3':np.empty_like(PT),'P6A':np.empty_like(PT),'P6B':np.empty_like(PT),'P9B':np.empty_like(PT),'P12B':np.empty_like(PT),'P12C':np.empty_like(PT),'P15B':np.empty_like(PT),'P18B':np.empty_like(PT),'sum':np.empty_like(PT)}
    D_dict = {'D3':np.empty_like(PT),'D6A':np.empty_like(PT),'D6B':np.empty_like(PT),'D9B':np.empty_like(PT),'D12B':np.empty_like(PT),'D12C':np.empty_like(PT),'D15B':np.empty_like(PT),'D18B':np.empty_like(PT),'Dz':np.empty_like(PT)}
    for num, protein in enumerate(PT):
        q = [protein,K1,K2,K3,K4,K5,K6,K7]
        ffs_partial = partial(ffs_complex,q)
        # Solutions are ordered according to how the initial guess vector is arranged
        solutions[num] = opt.root(ffs_partial,p,method='lm')
        p = [solutions[num].x[0],solutions[num].x[1],solutions[num].x[2],solutions[num].x[3],solutions[num].x[4],solutions[num].x[5],solutions[num].x[6],solutions[num].x[7]]
            
        P_dict['P3'][num] = 3*solutions[num].x[0]/protein
        P_dict['P6A'][num] = 6*solutions[num].x[1]/protein
        P_dict['P6B'][num] = 6*solutions[num].x[2]/protein
        P_dict['P9B'][num] = 9*solutions[num].x[3]/protein
        P_dict['P12B'][num] = 12*solutions[num].x[4]/protein
        P_dict['P12C'][num] = 12*solutions[num].x[5]/protein
        P_dict['P15B'][num] = 15*solutions[num].x[6]/protein
        P_dict['P18B'][num] = 18*solutions[num].x[7]/protein
        P_dict['sum'][num] = P_dict['P3'][num] + P_dict['P6A'][num] + P_dict['P6B'][num] + P_dict['P9B'][num] + P_dict['P12B'][num] \
        + P_dict['P12C'][num] + P_dict['P15B'][num] + P_dict['P18B'][num]
        
        D_dict['D3'][num] = kB*T/(6*np.pi*0.009982*5.0e-9)
        D_dict['D6A'][num] = D_dict['D3'][num]*2**(-0.15)
        D_dict['D6B'][num] = D_dict['D3'][num]*2**(-0.15)
        D_dict['D9B'][num] = D_dict['D3'][num]*3**(-0.333)
        D_dict['D12B'][num] = D_dict['D3'][num]*4**(-0.333)
        D_dict['D12C'][num] = D_dict['D3'][num]*4**(-0.333)
        D_dict['D15B'][num] = D_dict['D3'][num]*5**(-0.333)
        D_dict['D18B'][num] = D_dict['D3'][num]*6**(-0.333)
        D_dict['Dz'][num] = 
    
    return P_dict, D_dict


# Equations for dimer binding to two ligands sequentially with cooperativity, P2 + L <-> P2L + L <-> P2L2
def ffs_complex(q,p):
    
    # Unpack variables and constants
    M3, M6A, M6B, M9B, M12B, M12C, M15B, M18B = p # Variables
    PT, K1, K2, K3, K4, K5, K6, K7 = q # Constants
    
    # Equations have to be set up as equal to 0
    eq1 = -PT + 3*M3 + 6*M6A + 6*M6B + 9*M9B + 12*M12B + 12*M12C + 15*M15B + 18*M18B # Protein equation
    eq2 = 3*K1*M3**2 - M6A # Equilibrium constants
    eq3 = 3*K2*M3**2 - M6B
    eq4 = K3*M6B*M3 - M9B
    eq5 = 9*K4*M9B*M3 - M12B
    eq6 = (3/4)*K5*M9B*M3 - M12C
    eq7 = 3*K6*M12B*M3 - M15B
    eq8 = (1/2)*K7*M15B*M3 - M18B
    
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]


def plotting(PT, P_dict):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$P_{T}$ $\mu$M')
    ax.set_ylabel('Population')
    ax.set_ylim([-0.01,1.01])
    colors = ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e']

    plt.plot(PT*1e6,P_dict['P3'],color=colors[0],label='P3')
    plt.plot(PT*1e6,P_dict['P6A'],color=colors[1],label='P6A')
    plt.plot(PT*1e6,P_dict['P6B'],color=colors[2],label='P6B')
    plt.plot(PT*1e6,P_dict['P9B'],color=colors[3],label='P9B')
    plt.plot(PT*1e6,P_dict['P12B'],color=colors[4],label='P12B')
    plt.plot(PT*1e6,P_dict['P12C'],color=colors[5],label='P12C')
    plt.plot(PT*1e6,P_dict['P15B'],color=colors[6],label='P15B')
    plt.plot(PT*1e6,P_dict['P18B'],color=colors[7],label='P18B')
    plt.plot(PT*1e6,P_dict['sum'],'--',color='k',label='Sum')
    ax.legend()
        
    plt.show()
    
main()
        