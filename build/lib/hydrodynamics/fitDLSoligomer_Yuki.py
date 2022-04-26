#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:20:09 2019

@author: robertharkness
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
from lmfit import minimize, Parameters, report_fit

### Model for diffusion coefficients of trimer-hexamer system
### Assumes trimer diffusion can be treated as a linear function of concentration
def Trimer(fit_params, group_Tr):
    
    # Get temperatures and concentrations out of pandas groups
    Temperatures = np.array(group_Tr.Temperature) + 273.15
    Concentrations = np.array(group_Tr.Concentration*1e-6)

    # Initialize z-average diffusion coefficient array
    Dz_Tr = np.zeros(len(Concentrations))
    
    for x in range(len(Concentrations)):
        
        # Calculate ideal trimer diffusion coefficient
        D_Tr = StokesEinstein(Temperatures[x], fit_params['Rh_Tr'].value)
        
        # Calculate z-average diffusion coefficient
        Dz_Tr[x] = D_Tr*(1 + fit_params['k_c'].value*Concentrations[x])
    
    return Dz_Tr


### Model for diffusion coefficients of a trimer-hexamer system
def TrimerHexamer(fit_params, group_WT):
    
    # Set up arrays for calculating concentrations as a function of temperature and total monomer concentration
    Temperatures = np.array(group_WT.Temperature) + 273.15
    Concentrations = np.array(group_WT.Concentration)*1e-6
    
    Tr = np.zeros(len(Concentrations))
    H = np.zeros(len(Concentrations))
    
    # Numerator and denominator of diffusion coefficient equation
    Tr_num = np.zeros(len(Concentrations))
    H_num = np.zeros(len(Concentrations))
    Tr_den = np.zeros(len(Concentrations))
    H_den = np.zeros(len(Concentrations))
    Dz_TH = np.zeros(len(Concentrations))
    
    # Population dictionary for trimer and hexamer
    P_dict = {'PTr':[],'PH':[]}

    for x in range(len(Concentrations)):
            
    # Calculate equilibrium constants
        dG_H = fit_params['dH_H'].value - Temperatures[x]*fit_params['dS_H'].value
        K_H = np.exp(-dG_H/(R*Temperatures[x]))
            
    # Get free Trimer concentration from quadratic equation since this is 2Tr <-> Tr2 ie H
        Tr[x] = (-3 + np.sqrt(9 + 24*K_H*Concentrations[x]))/(12*K_H)    
        H[x] = (Concentrations[x] - 3*Tr[x])/6
        
        #print(f"CT: {Concentrations[x]}\nTemperature: {Temperatures[x]}\nTrimer concentration: {Tr[x]}\nHexamer concentration: {H[x]}")

    # Calculate the numerator and denominator terms and finally the average diffusion coefficient
        D_Tr = StokesEinstein(Temperatures[x], fit_params['Rh_Tr'].value)
        D_H = StokesEinstein(Temperatures[x], fit_params['Rh_H'].value)
            
        Tr_num[x] = Tr[x]*D_Tr*(1 + fit_params['k_c'].value*Tr[x])
        H_num[x] = 4*H[x]*D_H*(1 + fit_params['k_c'].value*H[x])
            
        Tr_den[x] = Tr[x]
        H_den[x] = 4*H[x]
            
        Dz_TH[x] = (Tr_num[x] + H_num[x])/(Tr_den[x] + H_den[x])
            
    # Make dictionary of the populations
        P_dict['PTr'].append(3*Tr[x]/Concentrations[x])
        P_dict['PH'].append(6*H[x]/Concentrations[x])
            
    return Dz_TH, P_dict


### Calculate diffusion coefficients
def StokesEinstein(T, Rh):
    
    kB = 1.38065e-23
    eta = (2.414e-5)*10**(247.8/(T-140))
    Dt = (kB*T)/(6*np.pi*eta*Rh)
    
    return Dt
   
    
### Minimization function for global fit
def objective(fit_params, data_Tr, data_WT):
    
    resid_Tr = []
    resid_TH = []
    for ind, group in data_Tr:
        Dz_Tr = Trimer(fit_params, group)*1e4
        for x,y in enumerate(Dz_Tr):
            resid_Tr.append((np.array(group.D.values[x]*1e4) - Dz_Tr[x]))
    
    for ind, group in data_WT:
        Dz_TH,P_dict = TrimerHexamer(fit_params, group)
        for x,y in enumerate(Dz_TH):
            resid_TH.append((np.array(group.D.values[x]*1e4) - Dz_TH[x]*1e4))
    
    resid = resid_Tr + resid_TH
    
    return resid


### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(data_Tr, data_WT, opt_params, CT_Trsim, CT_WTsim):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DLSglobalfits.pdf')
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    
    for ind, group in data_Tr:
        
        Tr_sim = pd.DataFrame()
        Tr_sim['Temperature'] = np.zeros(len(CT_Trsim))
        Tr_sim['Temperature'][:] = np.array(group['Temperature'])[0]
        Tr_sim['Concentration'] = CT_Trsim

        #Dz_Tr = Trimer(opt_params.params, Tr_sim)
        Dz_Tr = Trimer(opt_params, Tr_sim)
        Temperatures = np.array(group.Temperature)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(group.Concentration,group.D*1e4,'bo')
        ax.plot(Tr_sim['Concentration'],Dz_Tr*1e4,'r',LineWidth=2)
        ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        ax.yaxis.major.formatter._useMathText = True
        ax.set_title(f"DegP_5, Temperature: {np.round(Temperatures[0])} \N{DEGREE SIGN}C",fontsize=14)
        ax.set_xlabel('$[M]_T$ $\mu$M',fontsize=14)
        ax.set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$',fontsize=14)
        ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    
        pdf.savefig(fig)
        plt.close()
        
    for ind, group in data_WT:
        
        WT_sim = pd.DataFrame()
        WT_sim['Temperature'] = np.zeros(len(CT_WTsim))
        WT_sim['Temperature'] = np.array(group['Temperature'])[0]
        WT_sim['Concentration'] = CT_WTsim
        
        #Dz_TH, P_dict = TrimerHexamer(opt_params.params, WT_sim)
        Dz_TH, P_dict = TrimerHexamer(opt_params, WT_sim)
        Temperatures = np.array(group.Temperature)
        
        fig, axs = plt.subplots(1,2,figsize=(11,4))
        axs[0].plot(group.Concentration,group.D*1e4,'ko')
        axs[0].plot(WT_sim['Concentration'],Dz_TH*1e4,'r',LineWidth=2)
        axs[0].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        axs[0].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        axs[0].yaxis.major.formatter._useMathText = True
        axs[0].set_title(f"DegP_2, Temperature: {np.round(Temperatures[0])} \N{DEGREE SIGN}C",fontsize=14)
        axs[0].set_xlabel('$[M]_T$ $\mu$M',fontsize=14)
        axs[0].set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$',fontsize=14)
        axs[0].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        
        axs[1].plot(WT_sim['Concentration'],P_dict['PTr'],LineWidth=2,label='$P_{Tr}$')
        axs[1].plot(WT_sim['Concentration'],P_dict['PH'],LineWidth=2,label='$P_{H}$')
        axs[1].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        axs[1].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        axs[1].yaxis.major.formatter._useMathText = True
        axs[1].set_title(f"DegP_2, Temperature: {np.round(Temperatures[0])} \N{DEGREE SIGN}C",fontsize=14)
        axs[1].set_xlabel('$[M]_T$ $\mu$M',fontsize=14)
        axs[1].set_ylabel('Population',fontsize=14)
        axs[1].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        axs[1].legend(loc='upper left',fontsize=12,frameon=False)
        
        pdf.savefig(fig)
        plt.close()
        
    pdf.close()


### Read in data and set up for fitting
### Right now this script is set up to work on pandas dataframes but you can change it to work for your data format
def main():
    data = pd.read_csv(sys.argv[1])

    ### Set up parameters and groups    
    data_Tr = data[data['Sample'] == 'P5'] # DegP trimer mutant
    data_Tr = data_Tr[data_Tr['Temperature'] < 48]
    data_Tr = data_Tr[data_Tr['Concentration'] > 10]
    data_WT = data[data['Sample'] == 'P2'] # DegP protease-dead, WT
    data_WT = data_WT[data_WT['Temperature'] < 48]
    data_WT = data_WT[data_WT['Concentration'] > 10]
    
    groups_Tr = data_Tr.groupby('Temperature')
    groups_WT = data_WT.groupby('Temperature')
    
    CT_Trsim = np.linspace(np.max(data_Tr.Concentration),0.1,100)
    CT_WTsim = np.linspace(np.max(data_WT.Concentration),0.1,100)
    
    fit_params = Parameters()
    fit_params.add('dH_H',value=-300000)
    fit_params.add('dS_H',value=-900)
    fit_params.add('Rh_Tr',value=5.2e-9,min=2e-9,max=7e-9)
    fit_params.add('Rh_H',value=6.4e-9,min=3e-9,max=9e-9)
    fit_params.add('k_c',value=-1000,max=0)

    # Fit data
    #result = minimize(objective, fit_params, method='nelder', args=(groups_Tr,groups_WT))
    #report_fit(result)
    
    # Simulate data
    plot_fit(groups_Tr, groups_WT, fit_params, CT_Trsim, CT_WTsim)


### Run fit and generate result plots
R = 1.9872e-3 # Gas constant for calculating affinities
R = R*4.184*1000 # Thermodynamic parameters have to be in J/mol and J/mol*K to match this gas constant
main()



