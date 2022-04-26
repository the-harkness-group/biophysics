#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:48:51 2020

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

### Read in data and set up for fitting
def main():
    
    AUC_data = pd.read_csv(sys.argv[1])
    groups_WT = AUC_data.groupby('Temperature')
    
    fit_params = Parameters()    
    fit_params.add('dH6_0', value=-50000)
    fit_params.add('K6_0', value=756928.371)
    fit_params.add('dCp6', value=0)
    
    fit_params.add('dH12_0', value=61009.3724)
    fit_params.add('K12_0', value=402.618950)
    fit_params.add('dCp12', value=0)
    
    fit_params.add('S3', value=5.64, vary=False)
    fit_params.add('S6', value=10.2, vary=False)
    fit_params.add('S12', value=15.0, vary=False)
    
    # Fit data    
    result = minimize(objective, fit_params, method='nelder', args=(1, groups_WT))
    report_fit(result)
    opt_params = result.params
    plot_fit(groups_WT, opt_params)

# Thermodynamics
def P24(fit_params, group_WT):
    
    Temperatures = np.array(group_WT.Temperature + 273.15)
    Concentrations = np.array(group_WT.Concentration*1e-6)
    
    P_dict = {'P3':np.array([]),'c3':np.array([]),'P6':np.array([]),'c6':np.array([]),'P12':np.array([]),'c12':np.array([])}
    K_dict = {'K6':np.array([]),'K12':np.array([])}

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
        solutions = np.array([])
        p = [0.01,0.01,0.01]
        q = [Concentrations[y],K6,K12]
        ffs_partial = partial(ffs3,q)
        solutions = np.append(solutions,opt.root(ffs_partial,p,method='lm'))
        
        # Make dictionary of populations and concentrations
        for sol in solutions:
            P_dict['P3'] = np.append(P_dict['P3'],sol.x[0])
            P_dict['P6'] = np.append(P_dict['P6'],sol.x[1])
            P_dict['P12'] = np.append(P_dict['P12'],sol.x[2])
            
            P_dict['c3'] = np.append(P_dict['c3'],sol.x[0]*Concentrations[y]/3)
            P_dict['c6'] = np.append(P_dict['c6'],sol.x[1]*Concentrations[y]/6)
            P_dict['c12'] = np.append(P_dict['c12'],sol.x[2]*Concentrations[y]/12)
        
        # Make dictionary of the K values
        K_dict['K6'] = np.append(K_dict['K6'],K6)
        K_dict['K12'] = np.append(K_dict['K12'],K12)
        
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
    Sw_dict = {'Sw':np.array([])}
    Sw_dict['S3'] = np.zeros(len(P_dict['P3']))
    Sw_dict['S3'].fill(fit_params['S3'].value)
    Sw_dict['S6'] = np.zeros(len(P_dict['P6']))
    Sw_dict['S6'].fill(fit_params['S6'].value)
    Sw_dict['S12'] = np.zeros(len(P_dict['P12']))
    Sw_dict['S12'].fill(fit_params['S12'].value)
    
    Sw_dict['Sw'] = np.append(Sw_dict['Sw'],np.array(fit_params['S3'])*P_dict['P3'] + np.array(fit_params['S6'])*P_dict['P6'] + np.array(fit_params['S12'])*P_dict['P12'])
    
    return Sw_dict

# Minimization function
def objective(fit_params, x, data_WT):
    
    resid = np.array([])
    for ind, group in data_WT:
        P_dict,K_dict = P24(fit_params, group)
        Sw_dict = AUC(fit_params, P_dict)        
        resid = np.append(resid,group.Sw.values - Sw_dict['Sw'])
    
    #print(resid)
    
    return resid

### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(data_WT, opt_params):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DegP2_AUC_isothermfits.pdf')
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    colors = [plt.cm.rainbow(y) for y in range(500)]
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xscale('log')
    ax1.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    ax1.yaxis.major.formatter._useMathText = True
    #ax1.set_title(f"DegP_2, Temperature: {group.Temperature.iloc[0]} \N{DEGREE SIGN}C",fontsize=14)
    ax1.set_xlabel("Monomer concentration M",fontsize=14)
    ax1.set_ylabel('$S_{w}$',fontsize=14)
    ax1.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    ax2.yaxis.major.formatter._useMathText = True
    #ax2.set_title(f"DegP_2, Temperature: {group.Temperature.iloc[0]} \N{DEGREE SIGN}C",fontsize=14)
    ax2.set_xlabel("Monomer concentration $\mu$M",fontsize=14)
    ax2.set_ylabel('Population',fontsize=14)
    ax2.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    ax3.yaxis.major.formatter._useMathText = True
    #ax3.set_title(f"DegP_2, Temperature: {group.Temperature.iloc[0]} \N{DEGREE SIGN}C",fontsize=14)
    ax3.set_xlabel("Monomer concentration $\mu$M",fontsize=14)
    ax3.set_ylabel('Concentration $\mu$M',fontsize=14)
    ax3.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    
    color_idx = 150
    
    for ind, group in data_WT:
        
        sim_df = pd.DataFrame()
        #sim_df['Concentration'] = np.linspace(np.min(group.Concentration),np.max(group.Concentration),100)
        sim_df['Concentration'] = np.linspace(np.min(group.Concentration),896,100)
        sim_df['Temperature'] = np.zeros(len(sim_df['Concentration']))
        sim_df['Temperature'] = group.Temperature.iloc[0]
        P_dict, K_dict = P24(opt_params, sim_df)
        Sw_dict = AUC(opt_params, P_dict)
        
        ax1.plot(group.Concentration*1e-6,group.Sw,'o',color=colors[color_idx])
        ax1.plot(sim_df['Concentration']*1e-6,Sw_dict['Sw'],LineWidth=2,color=colors[color_idx])
        ax1.plot(sim_df['Concentration']*1e-6,Sw_dict['S3'],'--',LineWidth=2,color='#3288bd')
        ax1.plot(sim_df['Concentration']*1e-6,Sw_dict['S6'],'--',LineWidth=2,color='#99d594')
        ax1.plot(sim_df['Concentration']*1e-6,Sw_dict['S12'],'--',LineWidth=2,color='#d53e4f')

        ax2.plot(sim_df['Concentration'],P_dict['P3'],LineWidth=2,color='#3288bd')
        ax2.plot(sim_df['Concentration'],P_dict['P6'],LineWidth=2,color='#99d594')
        ax2.plot(sim_df['Concentration'],P_dict['P12'],LineWidth=2,color='#d53e4f')

        ax3.plot(sim_df['Concentration'],np.array(P_dict['c3'])*1e6,LineWidth=2,color='#3288bd')
        ax3.plot(sim_df['Concentration'],np.array(P_dict['c6'])*1e6,LineWidth=2,color='#99d594')
        ax3.plot(sim_df['Concentration'],np.array(P_dict['c12'])*1e6,LineWidth=2,color='#d53e4f')
        
        color_idx = color_idx-50
    
    #ax1.legend(loc='upper left',fontsize=12,frameon=False)
    #ax2.legend(loc='upper left',fontsize=12,frameon=False)
    #ax3.legend(loc='upper left',fontsize=12,frameon=False)
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.savefig(fig3)
    pdf.close()

R = 1.9872e-3 # Gas constant for calculating affinities
R = R*4.184*1000 # Thermodynamic parameters have to be in J/mol and J/mol*K to match this gas constant
T0 = 30+273.15
main()