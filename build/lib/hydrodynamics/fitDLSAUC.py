#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 20:31:58 2019

@author: robertharkness
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
from lmfit import minimize, Parameters, report_fit
from scipy import optimize as opt
from functools import partial
import time

### Read in data and set up for fitting
def main():
    
    AUC_data = pd.read_csv(sys.argv[1])
    #groups_WT = AUC_data.groupby('Temperature')
    
    fit_params = Parameters()    
    fit_params.add('dH6_0', value=-230000)
    fit_params.add('K6_0', value=278467)
    fit_params.add('dCp6', value=0,vary=False)
    
    fit_params.add('dH12_0', value=39284)
    fit_params.add('K12_0', value=2802)
    fit_params.add('dCp12', value=0,vary=False)
    
    #fit_params.add('S3', value=5.64, vary=False)
    fit_params.add('S3', value=6.241, vary=False)
    fit_params.add('S6', value=10.268, vary=False)
    fit_params.add('S12', value=15.0, vary=False)
    
    MC_iter = 100 # Number of Monte-Carlo error estimate iterations
    
    # Fit data    
    #result = minimize(objective, fit_params, method='nelder', args=(1, groups_WT))
    result = minimize(objective, fit_params, method='nelder', args=(1, AUC_data))
    report_fit(result)
    opt_params = result.params
    
    #start_time = time.time()
    RMSD = np.sqrt(result.chisqr/result.ndata)
    MC_dict, MC_data, errors = MonteCarloErrors(AUC_data, opt_params, RMSD, MC_iter)
    #print('Elapsed time: ',time.time() - start_time)
    
    print('\n')
    for k1,k2 in zip(opt_params.keys(),errors.keys()):
        print(f"{k1} = {opt_params[k1].value} +/- {errors[k2][0]}")
        
    plot_fit(AUC_data, opt_params, errors, MC_dict, MC_data)

# Thermodynamics
def P24(fit_params, data):
    
    Temperatures = np.array(data.Temperature + 273.15)
    Concentrations = np.array(data.Concentration*1e-6)
    
    P_dict = {'P3':[],'c3':[],'P6':[],'c6':[],'P12':[],'c12':[]}
    K_dict = {'K6':[],'K12':[]}

    dH6_0 = fit_params['dH6_0'].value
    K6_0 = abs(fit_params['K6_0'].value)
    dCp6 = fit_params['dCp6'].value
    
    dH12_0 = fit_params['dH12_0'].value
    K12_0 = abs(fit_params['K12_0'].value)
    dCp12 = fit_params['dCp12'].value

    for y in range(len(Concentrations)):
            
        # Calculate equilibrium constants
        K6 = K6_0*np.exp((dH6_0/R)*((1/T0) - (1/Temperatures[y])) + (dCp6/R)*(np.log(Temperatures[y]/T0) + (T0/Temperatures[y]) - 1))
        K12 = K12_0*np.exp((dH12_0/R)*((1/T0) - (1/Temperatures[y])) + (dCp12/R)*(np.log(Temperatures[y]/T0) + (T0/Temperatures[y]) - 1))
        
        # Get concentrations from solver
        solutions = []
        p = [0.01,0.01,0.01]
        q = [Concentrations[y],K6,K12]
        ffs_partial = partial(ffs3,q)
        solutions.append(opt.root(ffs_partial,p,method='lm'))
        
        # Make dictionary of populations and concentrations
        for sol in solutions:
            P_dict['P3'].append(sol.x[0])
            P_dict['P6'].append(sol.x[1])
            P_dict['P12'].append(sol.x[2])
            
            P_dict['c3'].append(sol.x[0]*Concentrations[y]/3)
            P_dict['c6'].append(sol.x[1]*Concentrations[y]/6)
            P_dict['c12'].append(sol.x[2]*Concentrations[y]/12)
        
        # Make dictionary of the K values
        K_dict['K6'].append(K6)
        K_dict['K12'].append(K12)
        
    return P_dict, K_dict

# Thermodynamics equation solver
def ffs3(q,p):
    
    CT, K6, K12 = q # Unpack constants
    P3, P6, P12 = p # Unpack variables
    
    eq1 = -1 + P3 + P6 + P12
    eq2 = K6*2*(P3**2)*CT - 3*P6
    eq3 = K12*(P6**2)*CT - 3*P12
    
    return [eq1, eq2, eq3]

# Generate AUC isotherms
def AUC(fit_params, P_dict):
    
    # Set up sedimentation coefficient dictionary
    Sw_dict = {'Sw':np.array([]),'S3':np.zeros(len(P_dict['P3'])),'S6':np.zeros(len(P_dict['P6'])),'S12':np.zeros(len(P_dict['P12']))}
    Sw_dict['S3'].fill(fit_params['S3'].value)
    Sw_dict['S6'].fill(fit_params['S6'].value)
    Sw_dict['S12'].fill(fit_params['S12'].value)
    
    Sw_dict['Sw'] = np.append(Sw_dict['Sw'],np.array(fit_params['S3'])*P_dict['P3'] + np.array(fit_params['S6'])*P_dict['P6'] + np.array(fit_params['S12'])*P_dict['P12'])
    
    return Sw_dict

# Minimization function
def objective(fit_params, x, data):

    resid = []
    P_dict,K_dict = P24(fit_params, data)
    Sw_dict = AUC(fit_params, P_dict)
    resid.append(data.Sw - Sw_dict['Sw'])

    return resid

### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(data, opt_params, errors, MC_dict, MC_data):
    
    data_groups = data.groupby('Temperature')
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DegP2_AUC_isothermfits.pdf')
    plt.style.use('figure')
    colors = ['#4575b4','#91bfdb','#fc8d59']
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xscale('log')
    #ax1.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    #ax1.yaxis.major.formatter._useMathText = True
    ax1.set_xlabel("$M_{T}$ $\mu$M")
    ax1.set_ylabel('$S_{w}$')
    #ax1.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    #ax1.text(0.40,0.85,"$\Delta H_{6}$ = %.2e $\pm$ %.2e $J$ $mol^{-1}$\n$K_{6}$ = %.2e $\pm$ %.2e $M^{-1}$\n$\Delta H_{12}$ = %.2e $\pm$ %.2e $J$ $mol^{-1}$\n$K_{12}$ = %.2e $\pm$ %.2e $M^{-1}$\
             #"%(opt_params['dH6_0'].value,errors['dH6_0 error'][0],opt_params['K6_0'].value,errors['K6_0 error'][0],\
             #opt_params['dH12_0'].value,errors['dH12_0 error'][0],opt_params['K12_0'].value,errors['K12_0 error'][0]),transform=ax1.transAxes,va="top")
    
    # Make dummy dataframe to get s-value limits
    sim_df = pd.DataFrame()
    sim_df['Concentration'] = np.linspace(np.min(data.Concentration),max(data.Concentration),1000)
    sim_df['Temperature'] = np.zeros(len(sim_df['Concentration']))
    sim_df['Temperature'] = 298.15
    P_dict, K_dict = P24(opt_params, sim_df)
    Sw_dict = AUC(opt_params, P_dict)
    ax1.tick_params(axis='x',pad=10)
    ax1.plot(sim_df['Concentration'],Sw_dict['S3'],'--',linewidth=2,color='#3288bd')
    ax1.plot(sim_df['Concentration'],Sw_dict['S6'],'--',linewidth=2,color='#99d594')
    ax1.plot(sim_df['Concentration'],Sw_dict['S12'],'--',linewidth=2,color='#d53e4f')
    ax1.text(0.85,0.9,"$M_{12}$",color='#d53e4f',fontsize=24,transform=ax1.transAxes,va="top")
    ax1.text(0.85,0.40,"$M_{6}$",color='#99d594',fontsize=24,transform=ax1.transAxes,va="top")
    ax1.text(0.85,0.2,"$M_{3}$",color='#3288bd',fontsize=24,transform=ax1.transAxes,va="top")
    
    # Make figure dictionary for storing figure handles
    #fig_dict = {f"fig_{x}":[] for x in range(1,len(data_groups)+1)}
    color_idx = 0
    fig_idx = 1
    
    for ind, group in data_groups:
        
        sim_df = pd.DataFrame()
        sim_df['Concentration'] = np.linspace(np.min(group.Concentration),np.max(group.Concentration),1000)
        #sim_df['Concentration'] = np.linspace(np.min(group.Concentration),896,1000)
        sim_df['Temperature'] = np.zeros(len(sim_df['Concentration']))
        sim_df['Temperature'] = group.Temperature.iloc[0]
        P_dict, K_dict = P24(opt_params, sim_df)
        Sw_dict = AUC(opt_params, P_dict)

        ax1.plot(group.Concentration,group.Sw,'o',color=colors[color_idx])
        ax1.plot(sim_df['Concentration'],Sw_dict['Sw'],linewidth=2,color=colors[color_idx])
        ax1.fill_between(sim_df['Concentration'],MC_data[ind]['Lower bound'],MC_data[ind]['Upper bound'],facecolor=colors[color_idx],edgecolor=colors[color_idx],alpha=0.1)
                 
        #fig2, ax2 = plt.subplots(1,2,figsize=(11,4))
        fig, ax = plt.subplots(1,1)
        #ax2[0].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
#        ax2[0].yaxis.major.formatter._useMathText = True
#        ax2[0].set_title(f"DegP_2, Temperature: {group.Temperature.iloc[0]} \N{DEGREE SIGN}C")
#        ax2[0].set_xlabel("Monomer concentration $\mu$M")
#        ax2[0].set_ylabel('Population')
#        ax2[0].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
#        ax2[0].plot(sim_df['Concentration'],P_dict['P3'],LineWidth=2,color='#3288bd',label='$P_{3}$')
#        ax2[0].plot(sim_df['Concentration'],P_dict['P6'],LineWidth=2,color='#99d594',label='$P_{6}$')
#        ax2[0].plot(sim_df['Concentration'],P_dict['P12'],LineWidth=2,color='#d53e4f',label='$P_{12}$')
#        ax2[0].legend(loc='upper left',frameon=False)
        
        ax.yaxis.major.formatter._useMathText = True
        ax.set_title(f"{group.Temperature.iloc[0]} \N{DEGREE SIGN}C",color=colors[color_idx])
        ax.set_xlabel("$M_{T}$ $\mu$M")
        ax.set_ylabel('Population')
        #ax2[0].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        ax.plot(sim_df['Concentration'],P_dict['P3'],linewidth=2,color='#3288bd',label='$P_{3}$')
        ax.plot(sim_df['Concentration'],P_dict['P6'],linewidth=2,color='#99d594',label='$P_{6}$')
        ax.plot(sim_df['Concentration'],P_dict['P12'],linewidth=2,color='#d53e4f',label='$P_{12}$')
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
        ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
        ax.set_ylim([-0.01,1.01])
        #ax.legend(loc='upper left',frameon=False)
    
        #ax2[1].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        #ax2[1].yaxis.major.formatter._useMathText = True
        #ax2[1].set_title(f"DegP_2, Temperature: {group.Temperature.iloc[0]} \N{DEGREE SIGN}C",fontsize=14)
        #ax2[1].set_xlabel("Monomer concentration $\mu$M")
        #ax2[1].set_ylabel('Concentration $\mu$M')
        #ax2[1].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        #ax2[1].plot(sim_df['Concentration'],np.array(P_dict['c3'])*1e6,LineWidth=2,color='#3288bd',label='$c_{3}$')
        #ax2[1].plot(sim_df['Concentration'],np.array(P_dict['c6'])*1e6,LineWidth=2,color='#99d594',label='$c_{6}$')
        #ax2[1].plot(sim_df['Concentration'],np.array(P_dict['c12'])*1e6,LineWidth=2,color='#d53e4f',label='$c_{12}$')
        #ax2[1].legend(loc='upper left',frameon=False)
                 
        #fig_dict[f"fig_{fig_idx}"] = fig2
        fig_idx = fig_idx + 1
        color_idx = color_idx + 1
        fig.tight_layout()
        fig.savefig(f"Populations_{ind}C.png",format='png')
        pdf.savefig(fig)
        plt.close(fig)
    
    # Monte-Carlo error figure
    #fig3, ax3 = plt.subplots(3,3,figsize=(11,6))
    #ax3 = ax3.ravel()
    #ks = [k for k in MC_dict.keys()]
    #for idx, ax in enumerate(ax3):
        #ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    #    ax.yaxis.major.formatter._useMathText = True
    #    ax.set_xlabel(f"Fitted {ks[idx]}")
    #    ax.set_ylabel('Count')
        #ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    #    ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    #    ax.hist(MC_dict[ks[idx]],bins=20,edgecolor='black',linewidth=1)
    #fig3.tight_layout()
    
    # Save figures
    fig1.tight_layout()
    fig1.savefig("Sw_withfits.png",format='png')
    pdf.savefig(fig1)
    plt.close(fig1)
    #for k,v in fig_dict.items():
    #    pdf.savefig(fig_dict[k])
#    pdf.savefig(fig3)
#    plt.close(fig3)
    pdf.close()
    
def MonteCarloErrors(data, opt_params, RMSD, MC_iter):
    
    perfect_data = data.copy()
    P_dict, K_dict = P24(opt_params, perfect_data)
    Sw_dict = AUC(opt_params, P_dict)
    perfect_data.Sw = Sw_dict['Sw']
    
    MC_dict = {k:[] for k in opt_params.keys()}
    errors = {k+' error':[] for k in MC_dict.keys()}
    MC_data = {}
    counter = 1
    for x in range(MC_iter):
        perturbed_data = perfect_data.copy()
        perturbed_data.Sw = perturbed_data.Sw + np.random.normal(scale=RMSD, size=np.size(perturbed_data.Sw))

        perturbed_result = minimize(objective, opt_params, method='nelder', args=(1, perturbed_data))
        {MC_dict[k].append(perturbed_result.params[k]) for k in perturbed_result.params.keys()}
        
        print(f"######### The current Monte-Carlo iteration is: {counter} #########")
        counter = counter + 1
     
    {errors[k+' error'].append(np.std(MC_dict[k])) for k in MC_dict.keys()}
    
    # Make simulated curves for plotting confidence intervals
    MC_data = {}
    perfect_groups = perfect_data.groupby('Temperature')
    for ind, group in perfect_groups: # Overwrite CSP values to be perfect values from optimal fit
        
        MC_data[ind] = {'Simulated data':[],'Upper bound':[],'Lower bound':[]}
        
        sim_df = pd.DataFrame()
        sim_df['Concentration'] = np.linspace(np.min(group.Concentration),np.max(group.Concentration),1000)
        #sim_df['Concentration'] = np.linspace(np.min(group.Concentration),896,1000)
        sim_df['Temperature'] = np.zeros(len(sim_df['Concentration']))
        sim_df['Temperature'] = group.Temperature.iloc[0]
        
        for x in range(MC_iter):
            
            MCsimparams = Parameters()
            
            MCsimparams.add('dH6_0', value=MC_dict['dH6_0'][x])
            MCsimparams.add('K6_0', value=MC_dict['K6_0'][x])
            MCsimparams.add('dCp6', value=0,vary=False)
    
            MCsimparams.add('dH12_0', value=MC_dict['dH12_0'][x])
            MCsimparams.add('K12_0', value=MC_dict['K12_0'][x])
            MCsimparams.add('dCp12', value=0,vary=False)
    
            #fit_params.add('S3', value=5.64, vary=False)
            MCsimparams.add('S3', value=6.241, vary=False)
            MCsimparams.add('S6', value=10.268, vary=False)
            MCsimparams.add('S12', value=15.0, vary=False)
            
            P_dict, K_dict = P24(MCsimparams, sim_df)
            Sw_dict = AUC(MCsimparams, P_dict)
                
            MC_data[ind]['Simulated data'].append(Sw_dict['Sw']) # Store data simulated from Monte Carlo parameters
        
        MC_data[ind]['Upper bound'] = np.mean(MC_data[ind]['Simulated data'],0) + 1.96*np.std(MC_data[ind]['Simulated data'],0) # Calculate upper and lower 95% confidence intervals
        MC_data[ind]['Lower bound'] = np.mean(MC_data[ind]['Simulated data'],0) - 1.96*np.std(MC_data[ind]['Simulated data'],0)
    

    return MC_dict, MC_data, errors

R = 1.9872e-3 # Gas constant for calculating affinities
R = R*4.184*1000 # Thermodynamic parameters have to be in J/mol and J/mol*K to match this gas constant
T0 = 30+273.15
main()