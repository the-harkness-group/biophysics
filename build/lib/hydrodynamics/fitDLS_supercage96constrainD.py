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

def supercage96(fit_params, group_WT):
    
    Temperatures = np.array(group_WT.Temperature + 273.15)
    Concentrations = np.array(group_WT.Concentration*1e-6)
    
    P_dict = {'P3':[],'c3':[],'P6':[],'c6':[],'P12':[],'c12':[],'P24':[],'c24':[],'P48':[],'c48':[],'P72':[],'c72':[],'P96':[],'c96':[]}
    D_dict = {'D3_0':[],'D6_0':[],'D12_0':[],'D24_0':[],'D48_0':[],'D72_0':[],'D96_0':[],'Dz':[]}
    K_dict = {'K6':[],'K12':[],'K24':[],'K48':[],'K72':[],'K96':[]}

    dH6_0 = fit_params['dH6_0'].value
    K6_0 = abs(fit_params['K6_0'].value)
    dCp6 = fit_params['dCp6'].value
    
    dH12_0 = abs(fit_params['dH12_0'].value)
    K12_0 = abs(fit_params['K12_0'].value)
    dCp12 = fit_params['dCp12'].value
    
    dH24_0 = abs(fit_params['dH24_0'].value)
    K24_0 = abs(fit_params['K24_0'].value)
    dCp24 = fit_params['dCp24'].value
    
    dH48_0 = abs(fit_params['dH48_0'].value)
    K48_0 = abs(fit_params['K48_0'].value)
    dCp48 = fit_params['dCp48'].value
    
    dH72_0 = abs(fit_params['dH72_0'].value)
    K72_0 = abs(fit_params['K72_0'].value)
    dCp72 = fit_params['dCp72'].value
    
    dH96_0 = abs(fit_params['dH96_0'].value)
    K96_0 = abs(fit_params['K96_0'].value)
    dCp96 = fit_params['dCp96'].value
    
    k_c = fit_params['k_c'].value
    
    M3 = 140438.4
    M6 = 2*M3
    M12 = 2*M6
    M24 = 2*M12
    M48 = 2*M24
    M72 = (3/2)*M48
    M96 = (4/3)*M72

    for y in range(len(Temperatures)):
            
        # Calculate equilibrium constants
        K6 = K6_0*np.exp((dH6_0/R)*((1/T0) - (1/Temperatures[y])) + (dCp6/R)*(np.log(Temperatures[y]/T0) + (T0/Temperatures[y]) - 1))
        K12 = K12_0*np.exp((dH12_0/R)*((1/T0) - (1/Temperatures[y])) + (dCp12/R)*(np.log(Temperatures[y]/T0) + (T0/Temperatures[y]) - 1))
        K24 = K24_0*np.exp((dH24_0/R)*((1/T0) - (1/Temperatures[y])) + (dCp24/R)*(np.log(Temperatures[y]/T0) + (T0/Temperatures[y]) - 1))
        K48 = K48_0*np.exp((dH48_0/R)*((1/T0) - (1/Temperatures[y])) + (dCp48/R)*(np.log(Temperatures[y]/T0) + (T0/Temperatures[y]) - 1))
        K72 = K72_0*np.exp((dH72_0/R)*((1/T0) - (1/Temperatures[y])) + (dCp72/R)*(np.log(Temperatures[y]/T0) + (T0/Temperatures[y]) - 1))
        K96 = K96_0*np.exp((dH96_0/R)*((1/T0) - (1/Temperatures[y])) + (dCp96/R)*(np.log(Temperatures[y]/T0) + (T0/Temperatures[y]) - 1))

        # Get concentrations from solver
        solutions = []
        p = [0.3,0.3,0.1,0.1,0.05,0.05,0.05]
        q = [Concentrations[0],K6,K12,K24,K48,K72,K96]
        ffs_partial = partial(ffs3,q)
        solutions.append(opt.root(ffs_partial,p,method='lm'))

        for sol in solutions:
            c3 = sol.x[0]*Concentrations[0]/3
            c6 = sol.x[1]*Concentrations[0]/6
            c12 = sol.x[2]*Concentrations[0]/12
            c24 = sol.x[3]*Concentrations[0]/24
            c48 = sol.x[4]*Concentrations[0]/48
            c72 = sol.x[5]*Concentrations[0]/72
            c96 = sol.x[6]*Concentrations[0]/96
        
        # Calculate the numerator and denominator terms and finally the average diffusion coefficient
        D3_0 = StokesEinstein(Temperatures[y], fit_params['Rh_3'].value)
        D6_0 = D3_0*2**(fit_params['n'].value)
        D12_0 = D3_0*4**(fit_params['n'].value)
        D24_0 = D3_0*8**(fit_params['n'].value)
        D48_0 = D3_0*16**(fit_params['n'].value)
        D72_0 = D3_0*24**(fit_params['n'].value)
        D96_0 = D3_0*32**(fit_params['n'].value)

        M3_num = (M3**2)*c3*(D3_0*(1 + k_c*Concentrations[0]))
        M6_num = (M6**2)*c6*(D6_0*(1 + k_c*Concentrations[0]))
        M12_num = (M12**2)*c12*(D12_0*(1 + k_c*Concentrations[0]))
        M24_num = (M24**2)*c24*(D24_0*(1 + k_c*Concentrations[0]))
        M48_num = (M48**2)*c48*(D48_0*(1 + k_c*Concentrations[0]))
        M72_num = (M72**2)*c72*(D72_0*(1 + k_c*Concentrations[0]))
        M96_num = (M96**2)*c96*(D96_0*(1 + k_c*Concentrations[0]))

        M3_den = (M3**2)*c3
        M6_den= (M6**2)*c6
        M12_den = (M12**2)*c12
        M24_den = (M24**2)*c24
        M48_den = (M48**2)*c48
        M72_den = (M72**2)*c72
        M96_den = (M96**2)*c96
        
        num_sum = (M3_num + M6_num + M12_num + M24_num + M48_num + M72_num + M96_num)
        den_sum = (M3_den + M6_den + M12_den + M24_den + M48_den + M72_den + M96_den)
        Dz = num_sum/den_sum
        
        # Make dictionary of the populations
        P_dict['P3'].append(3*c3/Concentrations[y])
        P_dict['P6'].append(6*c6/Concentrations[y])
        P_dict['P12'].append(12*c12/Concentrations[y])
        P_dict['P24'].append(24*c24/Concentrations[y])
        P_dict['P48'].append(48*c48/Concentrations[y])
        P_dict['P72'].append(72*c72/Concentrations[y])
        P_dict['P96'].append(96*c96/Concentrations[y])
        
        P_dict['c3'].append(c3)
        P_dict['c6'].append(c6)
        P_dict['c12'].append(c12)
        P_dict['c24'].append(c24)
        P_dict['c48'].append(c48)
        P_dict['c72'].append(c72)
        P_dict['c96'].append(c96)
        
        # Make dictionary of the D0 values
        D_dict['D3_0'].append(D3_0)
        D_dict['D6_0'].append(D6_0)
        D_dict['D12_0'].append(D12_0)
        D_dict['D24_0'].append(D24_0)
        D_dict['D48_0'].append(D48_0)
        D_dict['D72_0'].append(D72_0)
        D_dict['D96_0'].append(D96_0)
        D_dict['Dz'].append(Dz)
        
        # Make dictionary of the K values
        K_dict['K6'].append(K6)
        K_dict['K12'].append(K12)
        K_dict['K24'].append(K24)
        K_dict['K48'].append(K48)
        K_dict['K72'].append(K72)
        K_dict['K96'].append(K96)  
        
    return P_dict, K_dict, D_dict

def ffs3(q,p):
    
    CT, K6, K12, K24, K48, K72, K96 = q # Unpack constants
    P3, P6, P12, P24, P48, P72, P96 = p # Unpack variables
    
    eq1 = -1 + P3 + P6 + P12 + P24 + P48 + P72 + P96
    eq2 = K6*2*(P3**2)*CT - 3*P6
    eq3 = K12*(P6**2)*CT - 3*P12
    eq4 = K24*(P12**2)*CT - 6*P24
    eq5 = K48*(P24**2)*CT - 12*P48
    eq6 = K72*(P48*P24)*CT - 16*P72
    eq7 = K96*(P72*P24)*CT - 18*P96
    
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]

### Calculate diffusion coefficients
def StokesEinstein(T, Rh):
    
    kB = 1.38065e-23
    #eta_water = (2.414e-5)*10**(247.8/(T-140)) # Water Viscosity from Engineering ToolBox
    eta_HSBuffer = (9.46568e-11)*np.power(T,4) - (1.20155e-7)*np.power(T,3) + (5.73768e-5)*np.square(T) - (1.22266e-2)*T + 9.82422e-1 # HS DLS Buffer Viscosity from SEDNTERP1
    Dt = (kB*T)/(6*np.pi*eta_HSBuffer*Rh)
    
    return Dt
    
### Minimization function for global fit
def objective(fit_params, x, data_WT):
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    resid = []
    for ind, group in data_WT:
        P_dict,K_dict,D_dict = supercage96(fit_params, group)
        resid.append(group.D.values*1e4 - np.array(D_dict['Dz'])*1e4)
        
        #ax.plot(group.Temperature,group.D.values,'ko')
        #ax.plot(group.Temperature,D_dict['Dz'],'r')
        
    resid = np.ravel(resid)
    #print(resid,'\n')
    #ax.set_ylim(1.5e-11,8.5e-11)
    #plt.ion()
    #plt.pause(0.00001)
    #plt.ioff()
    #plt.close()
    
    return resid
    
### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(data_WT, opt_params):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DLSglobalfits.pdf')
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    colors = [plt.cm.rainbow(y) for y in range(500)]
    
    for ind, group in data_WT:
        
        sim_df = pd.DataFrame()
        sim_df['Temperature'] = np.linspace(np.min(group.Temperature),np.max(group.Temperature),100)
        sim_df['Concentration'] = np.zeros(len(sim_df['Temperature']))
        sim_df['Concentration'] = group.Concentration.iloc[0]
        P_dict, K_dict, D_dict = supercage96(opt_params, sim_df)
        
        D_fig = plt.figure()
        D_ax = D_fig.add_subplot(111)
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D3_0'])*1e4,color=colors[499],label='$D_{3,0}$')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D6_0'])*1e4,color=colors[199],label='$D_{6,0}$')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D12_0'])*1e4,color=colors[159],label='$D_{12,0}$')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D24_0'])*1e4,color=colors[119],label='$D_{24,0}$')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D48_0'])*1e4,color=colors[79],label='$D_{48,0}$')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D72_0'])*1e4,color=colors[39],label='$D_{72,0}$')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D96_0'])*1e4,color=colors[20],label='$D_{96,0}$')
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
        D_ax.legend(loc='upper left',fontsize=12,frameon=False)
        
        P_fig = plt.figure()
        P_ax = P_fig.add_subplot(111)
        P_ax.plot(sim_df['Temperature'],P_dict['P3'],color=colors[499],LineWidth=2,label='$P_{3}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P6'],color=colors[199],LineWidth=2,label='$P_{6}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P12'],color=colors[159],LineWidth=2,label='$P_{12}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P24'],color=colors[119],LineWidth=2,label='$P_{24}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P48'],color=colors[79],LineWidth=2,label='$P_{48}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P72'],color=colors[39],LineWidth=2,label='$P_{72}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P96'],color=colors[20],LineWidth=2,label='$P_{96}$')
        P_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        P_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        P_ax.yaxis.major.formatter._useMathText = True
        P_ax.set_title(f"DegP_2, $[M]_{{T}}$: {group.Concentration.iloc[0]} $\mu$M",fontsize=14)
        P_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
        P_ax.set_ylabel('Population',fontsize=14)
        P_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        P_ax.legend(loc='upper left',fontsize=12,frameon=False)
        
        c_fig = plt.figure()
        c_ax = c_fig.add_subplot(111)
        c_ax.plot(sim_df['Temperature'],np.array(P_dict['c3'])*1e6,color=colors[499],LineWidth=2,label='$c_{3}$')
        c_ax.plot(sim_df['Temperature'],np.array(P_dict['c6'])*1e6,color=colors[199],LineWidth=2,label='$c_{6}$')
        c_ax.plot(sim_df['Temperature'],np.array(P_dict['c12'])*1e6,color=colors[159],LineWidth=2,label='$c_{12}$')
        c_ax.plot(sim_df['Temperature'],np.array(P_dict['c24'])*1e6,color=colors[119],LineWidth=2,label='$c_{24}$')
        c_ax.plot(sim_df['Temperature'],np.array(P_dict['c48'])*1e6,color=colors[79],LineWidth=2,label='$c_{48}$')
        c_ax.plot(sim_df['Temperature'],np.array(P_dict['c72'])*1e6,color=colors[39],LineWidth=2,label='$c_{72}$')
        c_ax.plot(sim_df['Temperature'],np.array(P_dict['c96'])*1e6,color=colors[20],LineWidth=2,label='$c_{96}$')
        c_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        c_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        c_ax.yaxis.major.formatter._useMathText = True
        c_ax.set_title(f"DegP_2, $[M]_{{T}}$: {group.Concentration.iloc[0]} $\mu$M",fontsize=14)
        c_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
        c_ax.set_ylabel('Concentration $\mu$M',fontsize=14)
        c_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
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
    #data_WT = data_WT[data_WT['Temperature'] <= 30]
    #data_WT = data_WT[data_WT['Concentration'].isin([328.8, 258.3, 203, 159.5, 125.3, 98.5, 77.4, 60.8, 47.8, 37.5])]
    #data_WT = data_WT[data_WT['Concentration'].isin([862.7, 677.8, 532.6])]
    #data_WT.to_csv('DegP2_DLSparams_20190507.csv')
    #groups_WT = data_WT.groupby('Temperature')
    groups_WT = data_WT.groupby('Concentration')
    
    fit_params = Parameters()
    
    fit_params.add('dH6_0', value=-45070*4.184,vary=False)
    fit_params.add('K6_0', value=90909.09,vary=False)
    fit_params.add('dCp6', value=-2000)
    
    fit_params.add('dH12_0', value=218900.547)
    fit_params.add('K12_0', value=18729.3677)
    fit_params.add('dCp12', value=1000)
    
    fit_params.add('dH24_0', value=138934.776)
    fit_params.add('K24_0', value=20694.3490)
    fit_params.add('dCp24', value=1000)

    fit_params.add('dH48_0', value=8076.72055)
    fit_params.add('K48_0', value=11458.5678)
    fit_params.add('dCp48', value=1000)
    
    fit_params.add('dH72_0', value=147814.579)
    fit_params.add('K72_0', value=6007.87057)
    fit_params.add('dCp72', value=1000)

    fit_params.add('dH96_0', value=303318.894)
    fit_params.add('K96_0', value=14158.8488)
    fit_params.add('dCp96', value=1000)
    
    fit_params.add('Rh_3', value=5.0e-9,vary=False)
    fit_params.add('n',value=-0.333,vary=False)
    #fit_params.add('k_c', value=-52.56)
    fit_params.add('k_c',value=-389.669074)

    run_fit = 'y'
    
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
T0 = 30+273.15
main()

#    fit_params.add('dH12_0', value=180607.132)
#    fit_params.add('K12_0', value=27850.69)
#    fit_params.add('dCp12', value=0,vary=False)
    
#    fit_params.add('dH24_0', value=207018.863)
#    fit_params.add('K24_0', value=10710.3243)
#    fit_params.add('dCp24', value=0,vary=False)

#    fit_params.add('dH48_0', value=207018)
#    fit_params.add('K48_0', value=10000)
#    fit_params.add('dCp48', value=0,vary=False)
    
#    fit_params.add('dH72_0', value=207018.863)
#    fit_params.add('K72_0', value=10000)
#    fit_params.add('dCp72', value=0,vary=False)
    
#    fit_params.add('dH96_0', value=207018.863)
#    fit_params.add('K96_0', value=10000)
#    fit_params.add('dCp96', value=0,vary=False)