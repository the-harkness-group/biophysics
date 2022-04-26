#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:01:38 2019

@author: robertharkness
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import matplotlib.backends.backend_pdf
from lmfit import minimize, Parameters
import pandas as pd
from functools import partial

# Equations for trimer-hexamer equilibrium with binding of three ligands to trimer
def ffs_complex(q,p):
    
    # Unpack variables and constants
    T, TL, TL2, TL3, H, L = p # Variables
    CT, LT, KF, K1, K2, K3 = q # Constants
    
    # Equations have to be set up as equal to 0
    eq1 = -CT + 3*T + 3*TL + 3*TL2 + 3*TL3 + 6*H # Protein equation
    eq2 = -LT + L + TL + 2*TL2 + 3*TL3 # Ligand equation
    eq3 = KF*T**2 - H # Trimer-Hexamer equilibrium
    eq4 = K1*T*L - TL # Trimer binding first ligand
    eq5 = K2*TL*L - TL2 # Trimer binding second ligand
    eq6 = K3*TL2*L - TL3 # Trimer binding third ligand
    
    return [eq1, eq2, eq3, eq4, eq5, eq6]

# Calculate populations from the solver solutions
def Populations(CT,solutions):
    
    Pop_dict = {'T':[],'TL':[],'TL2':[],'TL3':[],'T_app':[],'Bound_app':[],'H':[]}
    for protein,sol in zip(CT,solutions):
        Pop_dict['T'].append(3*np.array(sol.x[0])/protein) # Trimer
        Pop_dict['TL'].append(3*np.array(sol.x[1])/protein) # Trimer 1-bound
        Pop_dict['TL2'].append(3*np.array(sol.x[2])/protein) # Trimer 2-bound
        Pop_dict['TL3'].append(3*np.array(sol.x[3])/protein) # Trimer 3-bound
        Pop_dict['H'].append(6*np.array(sol.x[4])/protein) # Hexamer
        Pop_dict['T_app'].append((3*np.array(sol.x[0])+2*np.array(sol.x[1])+np.array(sol.x[2]))/protein) # Apparent trimer population
        Pop_dict['Bound_app'].append((1*np.array(sol.x[1])+2*np.array(sol.x[2])+3*np.array(sol.x[3]))/protein) # Apparent bound population

    return Pop_dict

# Function for fitting, simulates data from fit parameters and compares to
# experimental data in minimization routine
def objective(fit_params,model,data):
    
    # Unpack parameters for fitting
    LT = data.Ligand*1e-6
    CT = data.Protein*1e-6
    KF = fit_params['KF']
    K1 = fit_params['K1']
    K2 = fit_params['K2']
    K3 = fit_params['K3']
    
    # Solve for unknown concentrations numerically using fsolve
    solutions = [] # List of solutions    
    for ligand,protein in zip(LT,CT):
        p = [protein,1e-6,1e-6,1e-6,1e-6,ligand]
        q = [protein,ligand,KF,K1,K2,K3]
        ffs_partial = partial(ffs_complex,q)
        # Solutions are ordered according to how the initial guess vector is arranged
        solutions.append(opt.root(ffs_partial,p,method='lm'))
    
    # Extract relevant concentrations from the solutions
    # Get populations for calculating residuals
    Pop_dict = Populations(CT,solutions)
    
    Trimer_resid = []
    Hexamer_resid = []
    Bound_resid = []
    for el,protein in enumerate(CT):
        Trimer_resid.append(Pop_dict['T_app'][el] - np.array(data.Trimer[el])) # Trimer signal is from all unbound trimer subunits
        Hexamer_resid.append(Pop_dict['H'][el] - np.array(data.Hexamer[el]))
        Bound_resid.append(Pop_dict['Bound_app'][el] - np.array(data.Bound[el])) # Bound signal is from all bound trimer subunits

    #print(f"Trimer residuals: {Trimer_resid}\n")
    #print(f"Hexamer residuals: {Hexamer_resid}\n")
    #print(f"Bound residuals: {Bound_resid}\n")
    resid = Trimer_resid + Hexamer_resid + Bound_resid
    print(f"Residuals:{resid}\nKF: {KF}\nK1: {K1}\nK2: {K2}\n")
    return resid

# Plot solver results and save pdf  
def plot_result(LT,CT,LT_sim,CT_sim,P_dict,params,data):
    
    # Set up PDF and plotting options for nice plots
    pdf = matplotlib.backends.backend_pdf.PdfPages("Yuki_NMRfits_20191119.pdf")
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2    
    colors = [plt.cm.rainbow(y) for y in range(500)]
    
    fig,ax = plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(LT/CT,data.Trimer,'o',color=colors[499],label='$P_{T,app.}$')
    ax[0].plot(LT/CT,data.Hexamer,'o',color=colors[209],label='$P_{H}$')
    ax[0].plot(LT/CT,data.Bound,'o',color=colors[49],label='$P_{Bound,app.}$')
    
    ax[0].plot(LT_sim/CT_sim,P_dict['T_app'],color=colors[499],label='$P_{T,app.}$') # T
#    ax.plot(LT/CT,P_dict['TL'],color=colors[399],label='$P_{TL_{3}}$') # TL
#    ax.plot(LT/CT,P_dict['TL2'],color=colors[299],label='$P_{TL_{3}}$') # TL2
#    ax.plot(LT/CT,P_dict['TL3'],color=colors[199],label='$P_{TL_{3}}$') # TL3
    ax[0].plot(LT_sim/CT_sim,P_dict['H'],color=colors[209],label='$P_{H}$') # H
    ax[0].plot(LT_sim/CT_sim,P_dict['Bound_app'],color=colors[49],label='$P_{Bound,app.}$') # Sum of bound populations
    ax[0].text(0.05,0.95,"$K_{F}$ = %.2e $M^{-1}$\n$K_{1}$ = %.2e $M^{-1}$\n$K_{2}$ = %.2e $M^{-1}$\n$K_{3}$ = %.2e $M^{-1}$"%(params['KF'],params['K1'],params['K2'],params['K3']),transform=ax[0].transAxes,va="top")
    ax[0].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    ax[0].yaxis.major.formatter._useMathText = True
    ax[0].set_xlabel('$L_{T}$/$C_{T}$',fontsize=14)
    ax[0].set_ylabel('Population',fontsize=14)
    ax[0].set_ylim(-0.05,1.05)
    ax[0].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    ax[0].legend(loc='upper right',frameon=False,fontsize=12)
    
    ax[1].plot(LT_sim/CT_sim,np.array(P_dict['TL'])+2*np.array(P_dict['TL2'])+3*np.array(P_dict['TL3']),color=colors[499]) # Average number of bound ligands
    ax[1].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    ax[1].yaxis.major.formatter._useMathText = True
    ax[1].set_xlabel('$L_{T}$/$C_{T}$',fontsize=14)
    ax[1].set_ylabel('Average ligands bound',fontsize=14)
    ax[1].set_ylim(-0.05,3.05)
    ax[1].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    
    pdf.savefig(fig)
    plt.show()
    pdf.close()

#########################################################################
##################### MAIN FUNCTION FITTING #############################
#########################################################################
def set_fit():
    
    # Read data to fit
    data = pd.read_csv('hTRA2_titration_20191117.csv')
    
    # Set up fit parameters
    fit_params = Parameters()
    #fit_params.add('KF',value=4.6e4,min=0) # Fixed KF since measured it
    #fit_params.add('K1',value=5e11,min=0)
    #fit_params.add('K2',value=5e25,min=0)
    
    # Other potential starting parameters
    fit_params.add('KF',value=4.6e4,min=0)
    fit_params.add('K1',value=1e6,min=0)
    fit_params.add('K2',value=1e5,min=0)
    fit_params.add('K3',value=1e4,min=0)

    model = 'complex' # Complex for trimer-hexamer equilibrium plus trimer binding three ligands sequentially
    
    ##########################################################################
    ############################## FITTING ###################################
    ##########################################################################
        
    # Experimental concentrations
    CT = np.array(data['Protein'])*1e-6
    LT = np.array(data['Ligand'])*1e-6
        
    # Run fit, pass in data containing concentrations and experimental populations
    result = minimize(objective, fit_params, args=(model,data))
    opt_params = result.params
        
    # Simulate data using optimized parameters
    # Solve for unknown concentrations numerically using fsolve
    LT_sim = np.linspace(1e-6,max(data.Ligand)*1e-6,100)
    CT_sim = np.zeros(len(LT_sim))
    CT_sim[:] = CT[0]
    solutions = [] # List of solutions
    for ligand,protein in zip(LT_sim,CT_sim):
        p = [protein,1e-6,1e-6,1e-6,1e-6,ligand]
        q = [protein,ligand,opt_params['KF'],opt_params['K1'],opt_params['K2'],opt_params['K3']]
        ffs_partial = partial(ffs_complex,q)
        # Solutions are ordered according to how the initial guess vector is arranged
        solutions.append(opt.root(ffs_partial,p,method='lm'))
    
    # Get populations
    Pop_dict = Populations(CT_sim,solutions)
        
    # Call plotting function with simulation/fit result
    plot_result(LT,CT,LT_sim,CT_sim,Pop_dict,opt_params,data)
        
set_fit()