#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:39:11 2020

@author: robertharkness
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import matplotlib.backends.backend_pdf
from lmfit import minimize, Parameters
from functools import partial


#########################################################################
##################### MAIN FUNCTION FITTING #############################
#########################################################################
def set_fit():
    
    ##########################################################################
    ############################## PARAMETERS ################################
    ##########################################################################
    
    # Set up simulation and fitting parameters
    fit_params = Parameters()
    fit_params.add('K',value=5000)
    alphas = np.array([0.1]) # Anti-, non-, positively-cooperative
    Models = ['Fast','Slow']
    
    # Experimental parameters
    expt_params = {}
    expt_params['PT'] = np.array(100e-6)
    expt_params['final_LT'] = np.array(5e-3)
    expt_params['npoints'] = 10 # Experimental titration points
    expt_params['noise_percent'] = 0.05 # gaussian noise
    expt_params['wU'] = 0
    expt_params['wB'] = 100
    expt_params['intensity'] = 1000
    expt_params['alphas'] = alphas
    # Generate experimental ligand titration points
    expt_params['LT'] = np.hstack([0,np.logspace(np.log10(0.05*expt_params['PT']),np.log10(expt_params['final_LT']),expt_params['npoints']-1,endpoint=True)])
    
    # Simulation parameters
    sim_params = {}
    sim_params['PT'] = expt_params['PT']
    sim_params['sim_points'] = 100
    sim_params['LT'] = np.hstack([0,np.logspace(np.log10(0.05*sim_params['PT']),np.log10(expt_params['final_LT']),sim_params['sim_points']-1,endpoint=True)])
    
    ##########################################################################
    ############################## SIMULATION ################################
    ##########################################################################
    
    # Iterate through models, cooperativities and simulate + fit data for dimer model
    # After dimer case for each model and cooperativity, simulate + fit P + L <-> PL model for comparison
    # Store results in dictionary for plotting at the end
    Plot_dict = {}
    for model in Models:
        
        expt_params['Model'] = model
        sim_params['Model'] = expt_params['Model']
    
        for index, alpha in enumerate(alphas):
        
            fit_params.add('alpha',value=alpha) # Cooperativity for dimer binding
        
            if expt_params['Model'] == 'Fast':
        
                fit_params.add('wU',value=expt_params['wU'])
                fit_params.add('wB',value=expt_params['wB'])
            
                # Simulate data according to dimer binding model, add noise
                sim_data, Pop_dict = NMRtitration(fit_params, expt_params)

                fast_noise = np.random.normal(0,expt_params['noise_percent']*np.mean(sim_data['Fast_exchange']),len(expt_params['LT']))
                expt_data = {'Fast_exchange':[]}
                expt_data['Fast_exchange'] = np.add(sim_data['Fast_exchange'],fast_noise)
            
            if expt_params['Model'] == 'Slow':
            
                fit_params.add('intensity',value=expt_params['intensity'])
            
                # Simulate data according to dimer binding model, add noise
                sim_data, Pop_dict = NMRtitration(fit_params, expt_params)
            
                slow_noise = np.random.normal(0,expt_params['noise_percent']*np.mean(sim_data['Slow_exchange']['U_app']),len(expt_params['LT']))
                expt_data = {'Slow_exchange':{'U_app':[],'B_app':[]}}
                expt_data['Slow_exchange']['U_app'] = np.add(sim_data['Slow_exchange']['U_app'],slow_noise)
                expt_data['Slow_exchange']['B_app'] = np.add(sim_data['Slow_exchange']['B_app'],slow_noise)

    ##########################################################################
    ############################## FITTING ###################################
    ##########################################################################
        
            # Run fit of simulated experimental data for dimer model using dimer model
            result = minimize(objective, fit_params, args=(expt_params,expt_data))
            opt_params = result.params
            #opt_params.pretty_print()
        
            # Simulate best fit data using optimized parameters for dimer model
            optsim_data, Pop_dict = NMRtitration(opt_params, sim_params)
            
            # Simulate best fit data for residual calculation
            resid_data, Pop_dict = NMRtitration(opt_params, expt_params)
        
            # Run fit of simulated experimental data for dimer model using P + L <-> PL model
            twostate_expt_params = expt_params.copy()
            twostate_expt_params['Model'] += ' P + L <-> PL'
            twostate_sim_params = sim_params.copy()
            twostate_sim_params['Model'] += ' P + L <-> PL'
        
            twostate_fit_params = Parameters()
            twostate_fit_params.add('K',value=1e4)
            if twostate_expt_params['Model'] == 'Slow P + L <-> PL':
                twostate_fit_params.add('intensity',value=1000)
            
            if twostate_expt_params['Model'] == 'Fast P + L <-> PL':
                twostate_fit_params.add('wU',value=0)
                twostate_fit_params.add('wB',value=100)
        
            twostate_result = minimize(objective, twostate_fit_params, args=(twostate_expt_params,expt_data))
            twostate_opt_params = twostate_result.params
            #twostate_opt_params.pretty_print()
        
            # Simulate best fit data using optimized parameters from P + L <-> PL fit to dimer data
            twostate_optsim_data, Pop_dict = NMRtitration(twostate_opt_params, twostate_sim_params)
            
            # Simulate best fit data using for P + L <-> PL for residual calculation
            twostate_resid_data, Pop_dict = NMRtitration(twostate_opt_params, twostate_expt_params)
            
            # Store data for plotting
            if expt_params['Model'] == 'Fast':
                
                Plot_dict[f"Fast_exchange_{alpha}"] = {}
                Plot_dict[f"Fast_exchange_{alpha}"]['expt_data'] = expt_data['Fast_exchange']
                Plot_dict[f"Fast_exchange_{alpha}"]['optsim_data'] = optsim_data['Fast_exchange']
                Plot_dict[f"Fast_exchange_{alpha}"]['residuals'] = expt_data['Fast_exchange'] - resid_data['Fast_exchange']
                
                Plot_dict[f"Twostate_Fast_exchange_{alpha}"] = {}
                Plot_dict[f"Twostate_Fast_exchange_{alpha}"]['optsim_data'] = twostate_optsim_data['Fast_exchange']
                Plot_dict[f"Twostate_Fast_exchange_{alpha}"]['residuals'] = expt_data['Fast_exchange'] - twostate_resid_data['Fast_exchange']

            if expt_params['Model'] == 'Slow':
                
                Plot_dict[f"Slow_exchange_{alpha}"] = {'U_app':{},'B_app':{}}
                Plot_dict[f"Slow_exchange_{alpha}"]['U_app']['expt_data'] = expt_data['Slow_exchange']['U_app']
                Plot_dict[f"Slow_exchange_{alpha}"]['U_app']['optsim_data'] = optsim_data['Slow_exchange']['U_app']
                Plot_dict[f"Slow_exchange_{alpha}"]['U_app']['residuals'] = expt_data['Slow_exchange']['U_app'] - resid_data['Slow_exchange']['U_app']
                
                Plot_dict[f"Twostate_Slow_exchange_{alpha}"] = {'U_app':{},'B_app':{}}
                Plot_dict[f"Twostate_Slow_exchange_{alpha}"]['U_app']['optsim_data'] = twostate_optsim_data['Slow_exchange']['U_app']
                Plot_dict[f"Twostate_Slow_exchange_{alpha}"]['U_app']['residuals'] = expt_data['Slow_exchange']['U_app'] - twostate_resid_data['Slow_exchange']['U_app']
                
                Plot_dict[f"Slow_exchange_{alpha}"]['B_app']['expt_data'] = expt_data['Slow_exchange']['B_app']
                Plot_dict[f"Slow_exchange_{alpha}"]['B_app']['optsim_data'] = optsim_data['Slow_exchange']['B_app']
                Plot_dict[f"Slow_exchange_{alpha}"]['B_app']['residuals'] = expt_data['Slow_exchange']['B_app'] - resid_data['Slow_exchange']['B_app']
                
                Plot_dict[f"Twostate_Slow_exchange_{alpha}"]['B_app']['optsim_data'] = twostate_optsim_data['Slow_exchange']['B_app']
                Plot_dict[f"Twostate_Slow_exchange_{alpha}"]['B_app']['residuals'] = expt_data['Slow_exchange']['B_app'] - twostate_resid_data['Slow_exchange']['B_app']
                
    ##########################################################################
    ############################## PLOTTING ##################################
    ##########################################################################

    # Call plotting function with simulation/fit result
    plot_result(Plot_dict, opt_params, expt_params, sim_params, twostate_sim_params)
        

# Calculate fast and slow exchange NMR titration observables
def NMRtitration(fit_params, expt_params):
    
    # Dimer binding
    if expt_params['Model'] == 'Fast' or expt_params['Model'] == 'Slow':
        # Unpack parameters for fitting
        LT = expt_params['LT']
        PT = expt_params['PT']
        K = np.abs(fit_params['K'].value)
        alpha = np.abs(fit_params['alpha'].value)
    
        # Solve for unknown concentrations numerically using fsolve
        solutions = [] # List of solutions   
        for ligand in LT:
            p = [PT,1e-6,1e-6,ligand]
            q = [PT,ligand,alpha,K]
            ffs_partial = partial(ffs_complex,q)
            # Solutions are ordered according to how the initial guess vector is arranged
            solutions.append(opt.root(ffs_partial,p,method='lm'))
    
        # Calculate fast or slow exchange observables in titration
        Pop_dict = Populations(PT,solutions,expt_params)
        if expt_params['Model'] == 'Fast':
        
            # Fast exchange
            NMR_dict = {'Fast_exchange':[]}
            NMR_dict['Fast_exchange'] = np.array(Pop_dict['U_app'])*fit_params['wU'].value + np.array(Pop_dict['B_app'])*fit_params['wB'].value
    
        if expt_params['Model'] == 'Slow':
    
            # Slow exchange
            NMR_dict = {'Slow_exchange':{'U_app':[],'B_app':[]}}
            NMR_dict['Slow_exchange']['U_app'] = np.array(Pop_dict['U_app'])*fit_params['intensity'].value
            NMR_dict['Slow_exchange']['B_app'] = np.array(Pop_dict['B_app'])*fit_params['intensity'].value
    
    # P + L <-> PL       
    if expt_params['Model'] == 'Fast P + L <-> PL' or expt_params['Model'] == 'Slow P + L <-> PL':
        
        # Unpack parameters for fitting
        LT = expt_params['LT']
        PT = expt_params['PT']
        K = np.abs(fit_params['K'].value)
        
        solutions = twostateinter(1./K, PT, LT)
        
        Pop_dict = Populations(PT,solutions,expt_params)
        if expt_params['Model'] == 'Fast P + L <-> PL':
            
            # Fast exchange
            NMR_dict = {'Fast_exchange':[]}
            NMR_dict['Fast_exchange'] = np.array(Pop_dict['U_app'])*fit_params['wU'].value + np.array(Pop_dict['B_app'])*fit_params['wB'].value
            
        if expt_params['Model'] == 'Slow P + L <-> PL':
            
            # Slow exchange
            NMR_dict = {'Slow_exchange':{'U_app':[],'B_app':[]}}
            NMR_dict['Slow_exchange']['U_app'] = np.array(Pop_dict['U_app'])*fit_params['intensity'].value
            NMR_dict['Slow_exchange']['B_app'] = np.array(Pop_dict['B_app'])*fit_params['intensity'].value
    
    return NMR_dict, Pop_dict


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


### Two-state P + L <-> PL
def twostateinter(Kd, PT, LT):
    
    a = (1/Kd)
    b = (1/Kd)*PT - (1/Kd)*LT + 1
    c = -LT
    
    L = (-b + np.sqrt(b**2 - 4.*a*c))/(2.*a)
    PL = LT - L
    P = PT - PL

    return [P, PL, L]


# Calculate populations from the solver solutions
def Populations(PT,solutions,expt_params):
    
    if expt_params['Model'] == 'Fast' or expt_params['Model'] == 'Slow':
        
        P2 = np.zeros(len(solutions))
        P2L = np.zeros(len(solutions))
        P2L2 = np.zeros(len(solutions))
        U_app = np.zeros(len(solutions))
        B_app = np.zeros(len(solutions))
        
        for index, sol in enumerate(solutions):
            
            P2[index] = sol.x[0]/PT # Unbound dimer
            P2L[index] = sol.x[1]/PT # 1-bound dimer
            P2L2[index] = sol.x[2]/PT # 2-bound dimer
            U_app[index] = (sol.x[0]  + 0.5*sol.x[1])/PT # Apparent unbound is sum of unbound protomers
            B_app[index] = (0.5*sol.x[1] + sol.x[2])/PT # Apparent bound is sum of bound protomers
            
        Pop_dict = {'P2':P2,'P2L':P2L,'P2L2':P2L2,'U_app':U_app,'B_app':B_app}
        
    if expt_params['Model'] == 'Fast P + L <-> PL' or expt_params['Model'] == 'Slow P + L <-> PL':
        
        Pop_dict = {'P':solutions[0]/PT,'PL':solutions[1]/PT,'U_app':solutions[0]/PT,'B_app':solutions[1]/PT} # U is P, B is PL
    
    return Pop_dict


# Function for fitting, simulates data from fit parameters and compares to
# experimental data in minimization routine
def objective(fit_params, expt_params, data):
    
    NMR_dict, Pop_dict = NMRtitration(fit_params, expt_params)
    
    resid = []
    if expt_params['Model'] == 'Fast':

        resid.append(NMR_dict['Fast_exchange'] - data['Fast_exchange']) # Only one observed average signal
        
    if expt_params['Model'] == 'Slow':

        resid.append(NMR_dict['Slow_exchange']['U_app'] - data['Slow_exchange']['U_app']) # Unbound signal is from all unbound dimer subunits
        resid.append(NMR_dict['Slow_exchange']['B_app'] - data['Slow_exchange']['B_app']) # Bound signal is from all bound dimer subunits
        
    if expt_params['Model'] == 'Fast P + L <-> PL':
        
        resid.append(NMR_dict['Fast_exchange'] - data['Fast_exchange'])
        
    if expt_params['Model'] == 'Slow P + L <-> PL':
        
        resid.append(NMR_dict['Slow_exchange']['U_app'] - data['Slow_exchange']['U_app']) # Unbound signal is from P
        resid.append(NMR_dict['Slow_exchange']['B_app'] - data['Slow_exchange']['B_app']) # Bound signal is from PL
    
    return resid


# Plot solver results and save pdf  
def plot_result(Plot_dict, opt_params, expt_params, sim_params, twostate_sim_params):
    
    # Set up PDF and plotting options for nice plots
    pdf = matplotlib.backends.backend_pdf.PdfPages("OligomerBinding_Fig2.pdf")
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 5
    #fastcolors = ['#fa9fb5','#dd3497','#7a0177'] # dark to light
    fastcolors = ['#7a0177','#dd3497','#fa9fb5']
    #slowcolors = ['#7fcdbb','#1d91c0','#253494'] # dark to light
    slowcolors = ['#253494','#1d91c0','#7fcdbb']
    widths = [3,3,3,3]
    heights = [3,1,3,1,1]
    
    # Set up plot
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axes = plt.subplots(ncols=4, nrows=5, constrained_layout=True,gridspec_kw=gs_kw,figsize=(50,30))
    for r, row in enumerate(axes):
        for c, ax in enumerate(row):
            ax.tick_params(direction='in',axis='both',length=10,width=5,grid_alpha=0.3,bottom=True,top=True,left=True,right=True,labelsize=36)
            ax.yaxis.major.formatter._useMathText = True
            ax.grid(linestyle='dashed',linewidth=5,dash_capstyle='round',dashes=(1,3))
            ax.yaxis.get_offset_text().set_fontsize(36)
            ax.yaxis.get_offset_text().set_fontweight('bold')
            
            #if (r == 0 and c == 0) or (r == 2 and c == 0):
            #    pass
            #else:
                #ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
            
            for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(36)
                item.set_fontweight('bold')

    # Plot titration results
    for index, alpha in enumerate(expt_params['alphas']):
        
        # Dimer fast exchange titration fits
        axes[0,1].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Fast_exchange_{alpha}"]['expt_data'],'o',markersize=14,color=fastcolors[index])
        axes[0,1].plot(sim_params['LT']/sim_params['PT'],Plot_dict[f"Fast_exchange_{alpha}"]['optsim_data'],linewidth=4,color=fastcolors[index])
        
        # Residuals
        axes[1,1].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Fast_exchange_{alpha}"]['residuals'],'o',markersize=14,color=fastcolors[index])
        
        # Two-state fast exchange titration fits
        axes[0,2].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Fast_exchange_{alpha}"]['expt_data'],'o',markersize=14,color=fastcolors[index])
        axes[0,2].plot(twostate_sim_params['LT']/twostate_sim_params['PT'],Plot_dict[f"Twostate_Fast_exchange_{alpha}"]['optsim_data'],'--',linewidth=4,color=fastcolors[index])
        
        # Residuals
        axes[1,2].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Twostate_Fast_exchange_{alpha}"]['residuals'],'o',markersize=14,color=fastcolors[index])
        
        # Dimer slow exchange titration fits
        axes[2,1].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Slow_exchange_{alpha}"]['U_app']['expt_data'],'o',markersize=14,color=slowcolors[index])
        axes[2,1].plot(sim_params['LT']/sim_params['PT'],Plot_dict[f"Slow_exchange_{alpha}"]['U_app']['optsim_data'],linewidth=4,color=slowcolors[index])
        axes[2,1].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Slow_exchange_{alpha}"]['B_app']['expt_data'],'d',markersize=14,color=slowcolors[index])
        axes[2,1].plot(sim_params['LT']/sim_params['PT'],Plot_dict[f"Slow_exchange_{alpha}"]['B_app']['optsim_data'],linewidth=4,color=slowcolors[index])

        # Residuals
        axes[3,1].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Slow_exchange_{alpha}"]['U_app']['residuals'],'o',markersize=14,color=slowcolors[index])
        axes[4,1].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Slow_exchange_{alpha}"]['B_app']['residuals'],'o',markersize=14,color=slowcolors[index])

        # Two-state slow exchange titration 
        axes[2,2].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Slow_exchange_{alpha}"]['U_app']['expt_data'],'o',markersize=14,color=slowcolors[index])
        axes[2,2].plot(twostate_sim_params['LT']/twostate_sim_params['PT'],Plot_dict[f"Twostate_Slow_exchange_{alpha}"]['U_app']['optsim_data'],'--',linewidth=4,color=slowcolors[index])
        axes[2,2].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Slow_exchange_{alpha}"]['B_app']['expt_data'],'d',markersize=14,color=slowcolors[index])
        axes[2,2].plot(twostate_sim_params['LT']/twostate_sim_params['PT'],Plot_dict[f"Twostate_Slow_exchange_{alpha}"]['B_app']['optsim_data'],'--',linewidth=4,color=slowcolors[index])
        
        # Residuals
        axes[3,2].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Twostate_Slow_exchange_{alpha}"]['U_app']['residuals'],'o',markersize=14,color=slowcolors[index])
        axes[4,2].plot(expt_params['LT']/expt_params['PT'],Plot_dict[f"Twostate_Slow_exchange_{alpha}"]['B_app']['residuals'],'o',markersize=14,color=slowcolors[index])
        
        # Axis properties
        axes[4,2].set_xlabel('$L_{T}$/$P_{T}$',fontsize=48,fontweight='bold')
        axes[0,1].set_title('$P_{2} + L$ $\\rightleftarrows$ $P_{2}L + L$ $\\rightleftarrows$ $P_{2}L_{2}$',fontsize=48,fontweight='bold')
        axes[0,2].set_title('$P + L$ $\\rightleftarrows$ $PL$',fontsize=48,fontweight='bold')
        
        axes[0,0].set_xlabel('$^{1}H$ ppm',fontsize=48,fontweight='bold')
        axes[0,0].set_ylabel('$^{15}N$ ppm',fontsize=48,fontweight='bold')
        axes[2,0].set_xlabel('$^{1}H$ ppm',fontsize=48,fontweight='bold')
        axes[2,0].set_ylabel('$^{15}N$ ppm',fontsize=48,fontweight='bold')
        axes[0,1].set_ylabel('Frequency',fontsize=48,fontweight='bold')
        axes[1,1].set_ylabel('Residual',fontsize=48,fontweight='bold')
        axes[2,1].set_ylabel('Intensity',fontsize=48,fontweight='bold')
        
        axes[0,0].set_ylim([130, 125])
        axes[0,0].set_xlim([8.5, 7.5])
        axes[2,0].set_ylim([130, 125])
        axes[2,0].set_xlim([8.5, 7.5])
        axes[0,2].set_yticklabels([])
        axes[2,2].set_yticklabels([])
        axes[1,1].set_xlim([-1,11])
        axes[3,1].set_xlim([-1,11])
        axes[3,1].set_xticklabels([])
        axes[4,1].set_xlim([-1,11])
        axes[1,2].set_xlim([-1,11])
        axes[3,2].set_xlim([-1,11])
        axes[3,2].set_xticklabels([])
        axes[4,2].set_xlim([-1,11])
        axes[1,0].axis('off')
        axes[3,0].axis('off')
        axes[4,0].axis('off')
        axes[2,1].set_ylim(-0.1*expt_params['intensity'],expt_params['intensity']+0.1*expt_params['intensity'])
        axes[0,1].set_ylim(-0.1*expt_params['wB'],expt_params['wB']+0.1*expt_params['wB'])
        

    pdf.savefig(fig)
    plt.savefig('Fig2.png',format='png')
    #plt.show()
    pdf.close()
    

set_fit()