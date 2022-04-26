#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:41:01 2019

@author: robertharkness
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import matplotlib.backends.backend_pdf
from lmfit import minimize, Parameters
import pandas as pd

# Equations for CT = 3*T + 3*TL3 + 6*H + 6*HL6 oligomer substrate binding
def ffs(p,*q):
    
    # Unpack variables and constants
    T, TL3, H, HL6, L = p # Variables
    CT, LT, KF, K1, K2 = q # Constants
    
    # Equations have to be set up as equal to 0
    eq1 = -CT + 3*T + 3*TL3 + 6*H + 6*HL6 # Protein equation
    eq2 = -LT + L + 3*TL3 + 6*HL6 # Ligand equation
    eq3 = KF*T**2 - H # Trimer-Hexamer equilibrium
    eq4 = K1*T*L**3 - TL3 # Trimer ligand binding
    eq5 = K2*H*L**6 - HL6 # Hexamer ligand binding
    
    return [eq1, eq2, eq3, eq4, eq5]

# Function for fitting, simulates data from fit parameters and compares to
# experimental data in minimization routine
def objective(fit_params,p,data):
    
    # Unpack parameters for fitting
    LT = data.Ligand*1e-6
    CT = data.Protein*1e-6
    KF = fit_params['KF']
    K1 = fit_params['K1']
    K2 = fit_params['K2']
    
    # Solve for unknown concentrations numerically using fsolve
    solutions = [] # List of solutions    
    for ligand,protein in zip(LT,CT):
        # Solutions are ordered according to how the initial guess vector is arranged
        solutions.append(opt.fsolve(ffs,p,(protein,ligand,KF,K1,K2)))
    
    # Extract relevant concentrations from the solutions
    Pop_dict = {'T':[],'TL3':[],'H':[],'HL6':[]}
    for protein,sol in zip(CT,solutions):
        Pop_dict['T'].append(3*np.array(sol[0])/protein)
        Pop_dict['TL3'].append(3*np.array(sol[1])/protein)
        Pop_dict['H'].append(6*np.array(sol[2])/protein)
        Pop_dict['HL6'].append(6*np.array(sol[3])/protein)
    
    Trimer_resid = []
    Hexamer_resid = []
    Bound_resid = []
    for el,protein in enumerate(CT):
        
        Trimer_resid.append(Pop_dict['T'][el] - np.array(data.Trimer[el]))
        Hexamer_resid.append(Pop_dict['H'][el] - np.array(data.Hexamer[el]))
        Bound_resid.append((Pop_dict['TL3'][el] + Pop_dict['HL6'][el]) - np.array(data.Bound[el])) # Bound signal is sum of TL3 and HL6

    print(f"Trimer residuals: {Trimer_resid}\n")
    print(f"Hexamer residuals: {Hexamer_resid}\n")
    print(f"Bound residuals: {Bound_resid}\n")
    resid = Trimer_resid + Hexamer_resid + Bound_resid
    print(f"Residuals:{resid}\nKF: {KF}\nK1: {K1}\nK2: {K2}\n")
    return resid

# Plot solver results and save pdf  
def plot_result(LT,CT,Concentration_dict,params,data,run_fit):
    
    # Unpack concentrations for plotting
    T = np.array(Concentration_dict['T'])
    TL3 = np.array(Concentration_dict['TL3'])
    H = np.array(Concentration_dict['H'])
    HL6 = np.array(Concentration_dict['HL6'])
    
    # Set up PDF and plotting options for nice plots
    pdf = matplotlib.backends.backend_pdf.PdfPages("Yuki_NMRsimulations.pdf")
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2    
    colors = [plt.cm.rainbow(y) for y in range(500)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if run_fit == 'y':
        ax.plot(LT/CT,data.Trimer,'o',color=colors[499])
        ax.plot(LT/CT,data.Hexamer,'o',color=colors[120])
        ax.plot(LT/CT,data.Bound,'o',color=colors[1])

    ax.plot(LT/CT,3*T/CT,color=colors[499],label='$P_{T}$') # T
    ax.plot(LT/CT,3*TL3/CT,color=colors[199],label='$P_{TL_{3}}$') # TL3
    ax.plot(LT/CT,6*H/CT,color=colors[120],label='$P_{H}$') # H
    ax.plot(LT/CT,6*HL6/CT,color=colors[45],label='$P_{HL_{6}}$') # HL3
    ax.plot(LT/CT,(6*HL6/CT)+(3*TL3/CT),color=colors[1],label='$P_{Bound}$') # Sum of bound populations
    ax.text(0.05,0.95,"$K_{F}$ = %.2e $M^{-1}$\n$K_{1}$ = %.2e $M^{-6}$\n$K_{2}$ = %.2e $M^{-6}$"%(params['KF'],params['K1'],params['K2']),transform=ax.transAxes,va="top")
    ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    ax.yaxis.major.formatter._useMathText = True
    ax.set_xlabel('$L_{T}$/$C_{T}$',fontsize=14)
    ax.set_ylabel('Population',fontsize=14)
    ax.set_ylim(-0.05,1.05)
    ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    ax.legend()
    pdf.savefig(fig)
    plt.show()
    pdf.close()

#########################################################################
############## MAIN FUNCTION FOR SIMULATING OR FITTING ##################
#########################################################################
def set_sim():
    
    # Read data to fit
    data = pd.read_csv('hTRA2_titration_20191117.csv')
    
    # Set up fit parameters
    fit_params = Parameters()
    fit_params.add('KF',value=4.6e4,min=0) # Fixed KF since measured it
    fit_params.add('K1',value=5e10,min=0)
    fit_params.add('K2',value=1e15,min=0)
    
    # Solve for unknown concentrations numerically using fsolve
    solutions = [] # List of solutions
    p = (10e-6,10e-6,10e-6,10e-6,10e-6) # Initial guesses for the solver [T, TL3, H, HL6, L]
    
    # Set option to simulate or fit, either 'y' or 'n'
    run_fit = 'y'
    
    ##########################################################################
    ############################## SIMULATIONS ###############################
    ##########################################################################
    
    if run_fit == 'n':

        # Experimental concentrations
        LT = np.logspace(-7,-3.5,50)
        CT = np.array(50e-6)
        
        for conc in LT:
            # Solutions are ordered according to how the initial guess vector is arranged
            solutions.append(opt.fsolve(ffs,p,(CT,conc,fit_params['KF'],fit_params['K1'],fit_params['K2'])))
    
        # Extract relevant concentrations from the solutions
        Concentration_dict = {'T':[],'TL3':[],'H':[],'HL6':[]}
        for sol in solutions:
            Concentration_dict['T'].append(sol[0])
            Concentration_dict['TL3'].append(sol[1])
            Concentration_dict['H'].append(sol[2])
            Concentration_dict['HL6'].append(sol[3])
        
        # Call plotting function with simulation/fit result
        plot_result(LT,CT,Concentration_dict,fit_params,data,run_fit)
    
    ##########################################################################
    ############################## FITTING ###################################
    ##########################################################################
    
    if run_fit == 'y':
        
        # Experimental concentrations
        CT = np.array(data['Protein'])*1e-6
        LT = np.array(data['Ligand'])*1e-6
        
        # Run fit, pass in data containing concentrations and experimental populations
        result = minimize(objective, fit_params, args=(p,data))
        opt_params = result.params
        
        # Simulate data using optimized parameters
        # Solve for unknown concentrations numerically using fsolve
        solutions = [] # List of solutions
        for ligand,protein in zip(LT,CT):
        # Solutions are ordered according to how the initial guess vector is arranged
            solutions.append(opt.fsolve(ffs,p,(protein,ligand,opt_params['KF'],opt_params['K1'],opt_params['K2'])))
    
        # Extract relevant concentrations from the solutions
        Concentration_dict = {'T':[],'TL3':[],'H':[],'HL6':[]}
        for sol in solutions:
            Concentration_dict['T'].append(sol[0])
            Concentration_dict['TL3'].append(sol[1])
            Concentration_dict['H'].append(sol[2])
            Concentration_dict['HL6'].append(sol[3])
        
        # Call plotting function with simulation/fit result
        plot_result(LT,CT,Concentration_dict,opt_params,data,run_fit)
        
set_sim()