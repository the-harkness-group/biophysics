#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:48:23 2020

@author: robertharkness
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import scipy as sp
import scipy.linalg
from functools import partial
import matplotlib


#########################################################################
##################### MAIN FUNCTION #####################################
#########################################################################
def main():
    
    ##########################################################################
    ############################## PARAMETERS ################################
    ##########################################################################
    
    # Simulation parameters, here rate constants are defined according to
    # k2 = alpha*k1, k-2 = k-1 for dimer binding
    # k1 is given by K*k-1
    sim_params = {}
    sim_params['PT'] = np.array(100e-6) # Protein concentration
    sim_params['LT'] = np.array(0.1*sim_params['PT']) # Ligand concentration
    sim_params['R2_U_P2'] = 10 # Transverse relaxation rates
    sim_params['R2_U_P2L'] = 10
    sim_params['R2_B_P2L'] = 10
    sim_params['R2_B_P2L2'] = 10
    sim_params['w_U_P2'] = 0 # Chemical shifts
    sim_params['w_U_P2L'] = 0
    sim_params['w_B_P2L'] = 200
    sim_params['w_B_P2L2'] = 200
    sim_params['K'] = 5000 # 200 uM Kd
    sim_params['alphas'] = np.array([10,1,0.1]) # Positive, non, anti cooperative
    sim_params['kminus1'] = 1000
    sim_params['k1'] = sim_params['K']*sim_params['kminus1']
    sim_params['Trelax'] = 0.04
    sim_params['CPMG_start'] = 50 # vcpmg parameters
    sim_params['CPMG_end'] = 1550
    sim_params['CPMG_step'] = 50

    ##########################################################################
    ############################## SIMULATION ################################
    ##########################################################################
    
    # Iterate through cooperativities and simulate CPMG profiles according to
    # the dimer binding to two ligands sequentially model
    sim_data, Pop_dict = CPMG(sim_params)

    ##########################################################################
    ############################## PLOTTING ##################################
    ##########################################################################

    # Call plotting function with simulation/fit result
    plot_result(sim_data, sim_params)
        

# Calculate CPMG dispersion profiles according to dimer binding model
def CPMG(sim_params):
    
    # Unpack and make CPMG frequency parameters
    Trelax = sim_params['Trelax']
    vcpmg = np.arange(sim_params['CPMG_start'],sim_params['CPMG_end'],sim_params['CPMG_step'])
    tcpmg = 0.25/vcpmg
    n = Trelax/(tcpmg*4.)
    
    # Make dictionary for storing simulated data
    NMR_dict = {}
    for alpha in sim_params['alphas']:
        
        NMR_dict[f"R2effU_{alpha}"] = np.zeros(len(vcpmg))
        NMR_dict[f"R2effB_{alpha}"] = np.zeros(len(vcpmg))
        
        # Get populations and propagation matrices
        Pop_dict = Populations(sim_params, alpha)
        Rp, Rm = MakeMatrices(sim_params, Pop_dict, alpha)
        
        # Initial conditions from dimer populations
        I0 = np.zeros((4,1),dtype=complex)
        I0[0,0] = Pop_dict['P2']
        I0[1,0] = 0.5*Pop_dict['P2L'] # P2L_U is 1/2 of P2L population
        I0[2,0] = 0.5*Pop_dict['P2L'] # P2L_B is 1/2 of P2L population
        I0[3,0] = Pop_dict['P2L2']

        # CPMG propagation
        R2effU = np.empty_like(vcpmg,dtype=float)
        R2effB = np.empty_like(vcpmg,dtype=float)

        for i in range(len(vcpmg)):
    
            Ucpmg = np.eye(4,dtype=complex)
    
            for k in range(int(n[i])): 
                
                Ucpmg = np.dot(sp.linalg.expm(Rp*tcpmg[i]),Ucpmg)
                Ucpmg = np.dot(sp.linalg.expm(Rm*tcpmg[i]),Ucpmg)
                Ucpmg = np.dot(sp.linalg.expm(Rm*tcpmg[i]),Ucpmg)
                Ucpmg = np.dot(sp.linalg.expm(Rp*tcpmg[i]),Ucpmg)
    
            It = np.dot(Ucpmg,I0)
            # R2eff for unbound magnetization terms calculated from P2_U and P2L_U
            R2effU[i] = (-1./Trelax)*np.log(np.real((It[0,0]+It[1,0])/(I0[0,0]+I0[1,0])))
            # R2eff for bound magnetization terms calculated from P2L_B and P2L2_B
            R2effB[i] = (-1./Trelax)*np.log(np.real((It[2,0]+It[3,0])/(I0[2,0]+I0[3,0])))
        
        NMR_dict[f"R2effU_{alpha}"] = R2effU
        NMR_dict[f"R2effB_{alpha}"] = R2effB
        NMR_dict[f"vcpmg_{alpha}"] = vcpmg
    
        print(f"########### Populations for alpha = {alpha} are (L is in uM not fractional population): #############")
        print(Pop_dict)
        print(f"kex for binding first ligand = {sim_params['k1']*Pop_dict['L'] + sim_params['kminus1']} per second")
        print(f"kex for binding second ligand = {alpha*sim_params['k1']*Pop_dict['L'] + sim_params['kminus1']} per second \n")
    
    return NMR_dict, Pop_dict


# Make Bloch-McConnell matrices for dimer binding model
def MakeMatrices(sim_params, Pop_dict, alpha):
    
    # Unpack CPMG parameters
    L = Pop_dict['L']
    k1 = sim_params['k1']
    kminus1 = sim_params['kminus1']
    k2 = alpha*sim_params['k1'] # k2 is alpha times k1
    kminus2 = sim_params['kminus1'] # k-2 = k-1
    w_U_P2 = sim_params['w_U_P2']
    w_U_P2L = sim_params['w_U_P2L']
    w_B_P2L = sim_params['w_B_P2L']
    w_B_P2L2 = sim_params['w_B_P2L2']
    R2_U_P2 = sim_params['R2_U_P2']
    R2_U_P2L = sim_params['R2_U_P2L']
    R2_B_P2L = sim_params['R2_B_P2L']
    R2_B_P2L2 = sim_params['R2_B_P2L2']
    
    # Relaxation and exchange matrix for dimer binding model
    # Non-explicitly set elements are zeros
    # Positive precession prior to 180 CPMG pulse
    Rp = np.zeros((4,4),dtype=complex)
    Rp[0,0] = 2*np.pi*w_U_P2*1j - R2_U_P2 - 2*k1*L
    Rp[0,1] = kminus1
    Rp[0,2] = kminus1
    Rp[1,0] = k1*L
    Rp[1,1] = 2*np.pi*w_U_P2L*1j - R2_U_P2L - kminus1 - k2*L
    Rp[1,3] = kminus2
    Rp[2,0] = k1*L
    Rp[2,2] = 2*np.pi*w_B_P2L*1j - R2_B_P2L - kminus1 - k2*L
    Rp[2,3] = kminus2
    Rp[3,1] = k2*L
    Rp[3,2] = k2*L
    Rp[3,3] = 2*np.pi*w_B_P2L2*1j - R2_B_P2L2 - 2*kminus2
    
    # Negative precession after 180 CPMG pulse
    Rm = np.zeros((4,4),dtype=complex)
    Rm[0,0] = -2*np.pi*w_U_P2*1j - R2_U_P2 - 2*k1*L
    Rm[0,1] = kminus1
    Rm[0,2] = kminus1
    Rm[1,0] = k1*L
    Rm[1,1] = -2*np.pi*w_U_P2L*1j - R2_U_P2L - kminus1 - k2*L
    Rm[1,3] = kminus2
    Rm[2,0] = k1*L
    Rm[2,2] = -2*np.pi*w_B_P2L*1j - R2_B_P2L - kminus1 - k2*L
    Rm[2,3] = kminus2
    Rm[3,1] = k2*L
    Rm[3,2] = k2*L
    Rm[3,3] = -2*np.pi*w_B_P2L2*1j - R2_B_P2L2 - 2*kminus2

    return Rp, Rm


# Calculate populations from the solver solutions
def Populations(sim_params, alpha):
    
    # Unpack parameters for fitting
    LT = sim_params['LT']
    PT = sim_params['PT']
    K = sim_params['K']
    
    # Solve for unknown concentrations numerically using fsolve
    solutions = [] # List of solutions   
    p = [PT,1e-6,1e-6,LT]
    q = [PT,LT,alpha,K]
    ffs_partial = partial(ffs_complex,q)
    # Solutions are ordered according to how the initial guess vector is arranged
    solutions.append(opt.root(ffs_partial,p,method='lm'))
    
    L = np.zeros(len(solutions))
    P2 = np.zeros(len(solutions))
    P2L = np.zeros(len(solutions))
    P2L2 = np.zeros(len(solutions))
    U_app = np.zeros(len(solutions))
    B_app = np.zeros(len(solutions))
        
    for index, sol in enumerate(solutions):
        
        P2[index] = sol.x[0]/PT # Unbound dimer
        P2L[index] = sol.x[1]/PT # 1-bound dimer
        P2L2[index] = sol.x[2]/PT # 2-bound dimer
        L[index] = sol.x[3] # Free ligand concentration
        U_app[index] = (sol.x[0]  + 0.5*sol.x[1])/PT # Apparent unbound is sum of unbound protomers
        B_app[index] = (0.5*sol.x[1] + sol.x[2])/PT # Apparent bound is sum of bound protomers
            
    Pop_dict = {'P2':P2,'P2L':P2L,'P2L2':P2L2,'U_app':U_app,'B_app':B_app,'L':L}

    return Pop_dict


# Equations for dimer binding to two ligands sequentially with cooperativity, P2 + L <-> P2L + L <-> P2L2
def ffs_complex(q,p):
    
    # Unpack variables and constants
    P2, P2L, P2L2, L = p # Variables
    PT, LT, alpha, K = q # Constants
    
    # Equations have to be set up as equal to 0
    eq1 = -PT + P2 + P2L + P2L2 # Protein equation
    eq2 = -LT + L + P2L + 2*P2L2 # Ligand equation
    eq3 = 2*K*P2*L - P2L
    eq4 = (1/2)*alpha*K*P2L*L - P2L2
    
    return [eq1, eq2, eq3, eq4]


# Plot CPMG results from dimer model and save figure
def plot_result(Plot_dict, sim_params):
    
    # Set up PDF and plotting options for nice plots
    label_params = {'mathtext.default': 'regular' }
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 3
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    CPMGcolors = ['#006d2c','#41ae76','#99d8c9']

    # Figure panels and axis labels
    fig1, ax1 = plt.subplots(1,figsize=(7,5))
    ax_list = [ax1]
    ax1.set_ylabel('$R_{2,eff.}$ [$s^{-1}$]',fontsize=24,fontweight='bold')
    ax1.set_xlabel('$\\it{{\\nu}}_{CPMG}$ [Hz]',fontsize=24,fontweight='bold')
    
    for c, ax in enumerate(ax_list):
        ax.tick_params(direction='in',axis='both',length=5,width=3,grid_alpha=0.3,bottom=True,top=True,left=True,right=True,labelsize=24)
        ax.yaxis.major.formatter._useMathText = True
        ax.yaxis.get_offset_text().set_fontsize(24)
        ax.yaxis.get_offset_text().set_fontweight('bold')

        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(24)
            item.set_fontweight('bold')
            
    # Plot simulation results
    text_shifts = [0.95 - 0.1*x for x in range(len(sim_params['alphas']))]
    for index, alpha in enumerate(sim_params['alphas']):

            ax1.plot(Plot_dict[f"vcpmg_{alpha}"],Plot_dict[f"R2effU_{alpha}"],linewidth=3,color=CPMGcolors[index])
            ax1.plot(Plot_dict[f"vcpmg_{alpha}"],Plot_dict[f"R2effB_{alpha}"],'--',linewidth=3,color=CPMGcolors[index])
            ax1.text(0.75,text_shifts[index],f"$\\it{{\\alpha}}$ = {alpha}",horizontalalignment='left', verticalalignment='top',transform=ax1.transAxes,fontsize=24,fontweight='bold',color=CPMGcolors[index])
    
    fig1.tight_layout()
    fig1.savefig('Fig3_CPMG.png',format='png')
    plt.close()

main()