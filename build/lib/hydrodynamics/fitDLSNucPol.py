#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:13:02 2019

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

def supercageNucPol(fit_params, group_WT):
    
    Temperatures = np.array(group_WT.Temperature + 273.15)
    Concentrations = np.array(group_WT.Concentration*1e-6)

    # Set up thermodynamic and hydrodynamic parameters
    dHn_0 = fit_params['dHn_0'].value
    Kn_0 = abs(fit_params['Kn_0'].value)
    dCpn = fit_params['dCpn'].value
    
    dHe_0 = fit_params['dHe_0'].value
    Ke_0 = abs(fit_params['Ke_0'].value)
    dCpe = fit_params['dCpe'].value
    
    k_d = fit_params['k_d'].value
    
    n = fit_params['n'].value
    N = int(fit_params['N'].value)
    m = fit_params['m'].value
    
    # Set up dictionaries for species up to size N
    P_dict = {}
    D_dict = {}
    c_dict = {}
    for x in range(1,N+1):
        P_dict[f"P{3*x}"] = []
        D_dict[f"D{3*x}_0"] = []
        c_dict[f"c{3*x}"] = []
    
    D_dict['Dz'] = []
    K_dict = {'Kn':[],'Ke':[]}

    for y in range(len(Temperatures)):
            
        # Calculate equilibrium constants
        Kn = Kn_0*np.exp((dHn_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpn/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        Ke = Ke_0*np.exp((dHe_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpe/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
       
        # Get concentrations from solver
        #solutions = []
        #p = [0.3]
        #q = [Concentrations[0],Kn,Ke]
        #ffs_partial = partial(ffs3,q)
        constants = [Concentrations[0]*Ke,Kn/Ke,n] # XT, sigma, nucleus size
        g = opt.fsolve(ffs3,1,args=constants)
        c3 = g/Ke
        
        # Generate populations and concentrations from solver solutions, calculate diffusion coefficients
        D_num = 0
        D_numtemp = 0
        D_den = 0
        D_dentemp = 0
        D3_0 = StokesEinstein(Temperatures[y], fit_params['Rh_3'].value)
        for x in range(1,N+1):
            
            if 3*x <= n:
                P_dict[f"P{3*x}"].append((3*x*Kn**(x-1)*c3**x)/Concentrations[0])
                c3x = Kn**(x-1)*c3**x
                c_dict[f"c{3*x}"].append(c3x)
                
            elif 3*x > n:
                P_dict[f"P{3*x}"].append((3*x*Kn**(n-1)*Ke**(x-n)*c3**x)/Concentrations[0])
                c3x = Kn**(n-1)*Ke**(x-n)*c3**x
                c_dict[f"c{3*x}"].append(c3x)
            
            # Calculate the numerator and denominator terms and finally the average diffusion coefficient
            D3x_0 = D3_0*x**(m)
            D_numtemp = (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
            D_num = D_num + D_numtemp
            D_dentemp = (x**2)*c3x
            D_den = D_den + D_dentemp
            D_dict[f"D{3*x}_0"].append(D3x_0)
        
        # Make dictionary of D values for fit
        Dz = D_num/D_den
        D_dict['Dz'].append(Dz)
        
        # Make dictionary of the K values
        K_dict['Kn'].append(Kn)
        K_dict['Ke'].append(Ke)
        
    return D_dict, P_dict, c_dict, K_dict

def ffs3(g,constants):
    
    XT, s, n = constants # Unpack constants
    
    eq1 = -XT + (1/s)*((3*g*s*(n*(g*s-1)*(g*s)**n-(g*s)**n+1))/(g*s-1)**2)+(s**(n-1))*((3*g**(n+1)*(-g*n+n+1))/(g-1)**2)
    
    return eq1

### Calculate diffusion coefficients
def StokesEinstein(T, Rh):
    
    kB = 1.38065e-23
    eta = (2.414e-5)*10**(247.8/(T-140)) # Water viscosity
    Dt = (kB*T)/(6*np.pi*eta*Rh)
    
    return Dt
    
### Minimization function for global fit
def objective(fit_params, x, data_WT):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    resid = []
    for ind, group in data_WT:
        D_dict, P_dict, c_dict, K_dict = supercageNucPol(fit_params, group)
        resid.append(group.D.values*1e4 - np.array(D_dict['Dz'])*1e4)
        
        ax.plot(group.Temperature,group.D.values,'ko')
        ax.plot(group.Temperature,D_dict['Dz'],'r')
        
    resid = np.ravel(resid)
    #print(resid,'\n')
    ax.set_ylim(1.5e-11,8.5e-11)
    plt.ion()
    plt.pause(0.00001)
    plt.ioff()
    plt.close()
    
    print(P_dict)
    #input('')
    
    return resid
    
### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(data_WT, opt_params):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DLSglobalfits.pdf')
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    colors = [plt.cm.rainbow(y) for y in range(100)]
    
    for ind, group in data_WT:
        
        sim_df = pd.DataFrame()
        sim_df['Temperature'] = np.linspace(np.min(group.Temperature),np.max(group.Temperature),100)
        sim_df['Concentration'] = np.zeros(len(sim_df['Temperature']))
        sim_df['Concentration'] = group.Concentration.iloc[0]
        D_dict, P_dict, c_dict, K_dict = supercageNucPol(opt_params, sim_df)
        
        D_fig = plt.figure()
        D_ax = D_fig.add_subplot(111)
        D_ax.plot(group.Temperature,group.D*1e4,'ko',label='Experiment')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['Dz'])*1e4,'r',label='Fit')
        D_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        D_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        D_ax.yaxis.major.formatter._useMathText = True
        D_ax.set_title(f"DegP_2, $[M]_{{T}}$: {group.Concentration.iloc[0]} $\mu$M",fontsize=14)
        D_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
        D_ax.set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$',fontsize=14)
        D_ax.set_ylim(min(group.D*1e4)-0.05*min(group.D*1e4),max(group.D*1e4)+0.05*max(group.D*1e4))
        D_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        
        P_fig = plt.figure()
        P_ax = P_fig.add_subplot(111)
        P_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        P_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        P_ax.yaxis.major.formatter._useMathText = True
        P_ax.set_title(f"DegP_2, $[M]_{{T}}$: {group.Concentration.iloc[0]} $\mu$M",fontsize=14)
        P_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
        P_ax.set_ylabel('Population',fontsize=14)
        P_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    
        c_fig = plt.figure()
        c_ax = c_fig.add_subplot(111)
        c_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        c_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        c_ax.yaxis.major.formatter._useMathText = True
        c_ax.set_title(f"DegP_2, $[M]_{{T}}$: {group.Concentration.iloc[0]} $\mu$M",fontsize=14)
        c_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
        c_ax.set_ylabel('Concentration $\mu$M',fontsize=14)
        c_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        
        for x in range(1,int(opt_params['N'].value)+1):
            D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}_0"])*1e4,color=colors[x],label=f"$D_{3*x}$")
            P_ax.plot(sim_df['Temperature'],P_dict[f"P{3*x}"],color=colors[x],LineWidth=2,label=f"$P_{3*x}$")        
            c_ax.plot(sim_df['Temperature'],np.array(c_dict[f"c{3*x}"])*1e6,color=colors[x],LineWidth=2,label=f"$c_{3*x}$")
        
        D_ax.legend(loc='upper left',fontsize=12,frameon=False)
        P_ax.legend(loc='upper left',fontsize=12,frameon=False)
        c_ax.legend(loc='upper left',fontsize=12,frameon=False)
        pdf.savefig(D_fig)
        pdf.savefig(P_fig)
        pdf.savefig(c_fig)
        plt.close(D_fig)
        plt.close(P_fig)
        plt.close(c_fig)
        
    pdf.close()
    
### Read in data and set up for fitting
def main():
    data = pd.read_csv(sys.argv[1])

    ### Set up parameters and groups    
    data_WT = data[data['Sample'] == 'P2'] # DegP protease-dead, WT
    #data_WT = data_WT[data_WT['Concentration'] < 150]
    data_WT = data_WT[data_WT['Temperature'] <= 30]
    #data_WT = data_WT[data_WT['Concentration'].isin([328.8, 258.3, 203, 159.5, 125.3, 98.5, 77.4, 60.8, 47.8, 37.5])]
    #data_WT.to_csv('DegP2_DLSparams_20190507.csv')
    #groups_WT = data_WT.groupby('Temperature')
    groups_WT = data_WT.groupby('Concentration')
    
    fit_params = Parameters()
    
    fit_params.add('dHn_0', value=-45070*4.184,vary=False)
    fit_params.add('Kn_0', value=90909.09,vary=False)
    fit_params.add('dCpn', value=0,vary=False)
    
    fit_params.add('dHe_0', value=180607.132)
    fit_params.add('Ke_0', value=27850.69)
    fit_params.add('dCpe', value=0,vary=False)
    
#    fit_params.add('dH24_0', value=0,vary=False)
#    fit_params.add('K24_0', value=0,vary=False)
#    fit_params.add('dCp24', value=0,vary=False)

    fit_params.add('T0', value=30+273.15, vary=False)
    
    fit_params.add('Rh_3', value=5e-9)
    fit_params.add('m',value=-1/3)
    fit_params.add('n',value=4)
    fit_params.add('N',value=32)
    #fit_params.add('k_c', value=-52.56)
    fit_params.add('k_d',value=1.05)

    run_fit = 'n'
    
    if run_fit == 'y':
        # Fit data    
        result = minimize(objective, fit_params, method='nelder', args=(1, groups_WT))
        report_fit(result)
        opt_params = result.params
        plot_fit(groups_WT, opt_params)
        #params_df = pd.DataFrame(opt_params)
        #params_df.to_csv('optimized_fit_params.csv')
        
    if run_fit == 'n':
        # Simulate data
        plot_fit(groups_WT, fit_params)

R = 1.9872e-3 # Gas constant for calculating affinities
R = R*4.184*1000 # Thermodynamic parameters have to be in J/mol and J/mol*K to match this gas constant
main()