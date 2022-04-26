#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:33:24 2020

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

def P24(fit_params, group_WT):
    
    Temperatures = np.array(group_WT.Temperature + 273.15)
    Concentrations = np.array(group_WT.Concentration*1e-6)
    
    P_dict = {'P3':[],'c3':[],'P6':[],'c6':[],'P12':[],'c12':[],'P24':[],'c24':[]}
    D_dict = {'Dz':[],'D3_0':[],'D3':[],'D6_0':[],'D6':[],'D12_0':[],'D12':[],'D24_0':[],'D24':[]}
    K_dict = {'K6':[],'K12':[],'K24':[]}

    dH6_0 = fit_params['dH6_0'].value
    K6_0 = abs(fit_params['K6_0'].value)
    dCp6 = fit_params['dCp6'].value
    
    dH12_0 = fit_params['dH12_0'].value
    K12_0 = abs(fit_params['K12_0'].value)
    dCp12 = fit_params['dCp12'].value
    
    dH24_0 = fit_params['dH24_0'].value
    K24_0 = abs(fit_params['K24_0'].value)
    dCp24 = fit_params['dCp24'].value
    
    kc3 = fit_params['kc3'].value
    kc6 = fit_params['kc6'].value
    kc12 = fit_params['kc12'].value
    kc24 = fit_params['kc24'].value
    
    M3 = 140438.4
    M6 = 2*M3
    M12 = 2*M6
    M24 = 2*M12

    for y in range(len(Temperatures)):
            
        # Calculate equilibrium constants
        K6 = K6_0*np.exp((dH6_0/R)*((1/T0) - (1/Temperatures[y])) + (dCp6/R)*(np.log(Temperatures[y]/T0) + (T0/Temperatures[y]) - 1))
        K12 = K12_0*np.exp((dH12_0/R)*((1/T0) - (1/Temperatures[y])) + (dCp12/R)*(np.log(Temperatures[y]/T0) + (T0/Temperatures[y]) - 1))
        K24 = K24_0*np.exp((dH24_0/R)*((1/T0) - (1/Temperatures[y])) + (dCp24/R)*(np.log(Temperatures[y]/T0) + (T0/Temperatures[y]) - 1))

        # Get concentrations from solver
        solutions = []
        #p = [0.3,0.3,0.1,0.1]
        p = [0.01,0.01,0.01,0.01]
        q = [Concentrations[0],K6,K12,K24]
        ffs_partial = partial(ffs3,q)
        solutions.append(opt.root(ffs_partial,p,method='lm'))

        for sol in solutions:
            c3 = sol.x[0]*Concentrations[0]/3
            c6 = sol.x[1]*Concentrations[0]/6
            c12 = sol.x[2]*Concentrations[0]/12
            c24 = sol.x[3]*Concentrations[0]/24
        
        # Calculate the numerator and denominator terms and finally the average diffusion coefficient
        D3_0 = StokesEinstein(Temperatures[y], fit_params['Rh_3'].value)
        D6_0 = D3_0*2**(fit_params['n'].value)
        D12_0 = D3_0*4**(fit_params['n'].value)
        D24_0 = D3_0*8**(fit_params['n'].value)

        M3_num = (M3**2)*c3*(D3_0*(1 + kc3*Concentrations[0]))
        M6_num = (M6**2)*c6*(D6_0*(1 + kc6*Concentrations[0]))
        M12_num = (M12**2)*c12*(D12_0*(1 + kc12*Concentrations[0]))
        M24_num = (M24**2)*c24*(D24_0*(1 + kc24*Concentrations[0]))

        M3_den = (M3**2)*c3
        M6_den= (M6**2)*c6
        M12_den = (M12**2)*c12
        M24_den = (M24**2)*c24
        
        num_sum = (M3_num + M6_num + M12_num + M24_num)
        den_sum = (M3_den + M6_den + M12_den + M24_den)
        Dz = num_sum/den_sum
        
        # Make dictionary of the populations
        P_dict['P3'].append(3*c3/Concentrations[y])
        P_dict['P6'].append(6*c6/Concentrations[y])
        P_dict['P12'].append(12*c12/Concentrations[y])
        P_dict['P24'].append(24*c24/Concentrations[y])
        
        P_dict['c3'].append(c3)
        P_dict['c6'].append(c6)
        P_dict['c12'].append(c12)
        P_dict['c24'].append(c24)
        
        # Make dictionary of the D0 values
        D_dict['D3_0'].append(D3_0)
        D_dict['D6_0'].append(D6_0)
        D_dict['D12_0'].append(D12_0)
        D_dict['D24_0'].append(D24_0)
        D_dict['D3'].append(D3_0*(1 + kc3*Concentrations[0]))
        D_dict['D6'].append(D6_0*(1 + kc6*Concentrations[0]))
        D_dict['D12'].append(D12_0*(1 + kc12*Concentrations[0]))
        D_dict['D24'].append(D24_0*(1 + kc24*Concentrations[0]))
        D_dict['Dz'].append(Dz)
        
        # Make dictionary of the K values
        K_dict['K6'].append(K6)
        K_dict['K12'].append(K12)
        K_dict['K24'].append(K24)
        
    return P_dict, K_dict, D_dict

def ffs3(q,p):
    
    CT, K6, K12, K24 = q # Unpack constants
    P3, P6, P12, P24 = p # Unpack variables
    
    eq1 = -1 + P3 + P6 + P12 + P24
    eq2 = K6*2*(P3**2)*CT - 3*P6
    eq3 = K12*(P6**2)*CT - 3*P12
    eq4 = K24*(P12**2)*CT - 6*P24
    
    return [eq1, eq2, eq3, eq4]

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
        P_dict,K_dict,D_dict = P24(fit_params, group)
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
    print('RSS: ',np.sum(np.square(resid)))
    
    return resid
    
### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(data_WT, opt_params):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DLSglobalfits.pdf')
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    colors = [plt.cm.rainbow(y) for y in range(500)]
    
    D_all_fig = plt.figure()
    D_all_ax = D_all_fig.add_subplot(111)
    D_all_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    D_all_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    D_all_ax.yaxis.major.formatter._useMathText = True
    D_all_ax.set_title("DegP_2, all concentrations",fontsize=14)
    D_all_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
    D_all_ax.set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$',fontsize=14)
    D_all_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    
    for ind, group in data_WT:
        
        sim_df = pd.DataFrame()
        sim_df['Temperature'] = np.linspace(np.min(group.Temperature),np.max(group.Temperature),100)
        sim_df['Concentration'] = np.zeros(len(sim_df['Temperature']))
        sim_df['Concentration'] = group.Concentration.iloc[0]
        P_dict, K_dict, D_dict = P24(opt_params, sim_df)
        
        D_fig = plt.figure()
        D_ax = D_fig.add_subplot(111)
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D3_0'])*1e4,color=colors[499],label='$D_{3,0}$')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D6_0'])*1e4,color=colors[199],label='$D_{6,0}$')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D12_0'])*1e4,color=colors[159],label='$D_{12,0}$')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D24_0'])*1e4,color=colors[119],label='$D_{24,0}$')
        if group.Concentration.iloc[0] == 862.7:
            D_ax.plot(sim_df['Temperature'],np.array(D_dict['D3'])*1e4,'--',color=colors[499],label='$D(C_{T})_{3}$')
            D_ax.plot(sim_df['Temperature'],np.array(D_dict['D6'])*1e4,'--',color=colors[199],label='$D(C_{T})_{6}$')
            D_ax.plot(sim_df['Temperature'],np.array(D_dict['D12'])*1e4,'--',color=colors[159],label='$D(C_{T})_{12}$')
            D_ax.plot(sim_df['Temperature'],np.array(D_dict['D24'])*1e4,'--',color=colors[119],label='$D(C_{T})_{24}$')
        D_ax.plot(group.Temperature,group.D*1e4,'ko',label='Experiment')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['Dz'])*1e4,'C9',label='Fit')
        D_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        D_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        D_ax.yaxis.major.formatter._useMathText = True
        D_ax.set_title(f"DegP_2, $[M]_{{T}}$: {group.Concentration.iloc[0]} $\mu$M",fontsize=14)
        D_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
        D_ax.set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$',fontsize=14)
        D_ax.set_ylim(min(group.D*1e4)-0.05*min(group.D*1e4),max(group.D*1e4)+0.05*max(group.D*1e4))
        D_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        D_ax.legend(loc='upper left',fontsize=12,frameon=False)

        D_all_ax.plot(sim_df['Temperature'],np.array(D_dict['D3_0'])*1e4,color=colors[499],label='$D_{3,0}$')
        D_all_ax.plot(sim_df['Temperature'],np.array(D_dict['D6_0'])*1e4,color=colors[199],label='$D_{6,0}$')
        D_all_ax.plot(sim_df['Temperature'],np.array(D_dict['D12_0'])*1e4,color=colors[159],label='$D_{12,0}$')
        D_all_ax.plot(sim_df['Temperature'],np.array(D_dict['D24_0'])*1e4,color=colors[119],label='$D_{24,0}$')
        if group.Concentration.iloc[0] == 862.7:
            D_all_ax.plot(sim_df['Temperature'],np.array(D_dict['D3'])*1e4,'--',color=colors[499],label='$D(C_{T})_{3}$')
            D_all_ax.plot(sim_df['Temperature'],np.array(D_dict['D6'])*1e4,'--',color=colors[199],label='$D(C_{T})_{6}$')
            D_all_ax.plot(sim_df['Temperature'],np.array(D_dict['D12'])*1e4,'--',color=colors[159],label='$D(C_{T})_{12}$')
            D_all_ax.plot(sim_df['Temperature'],np.array(D_dict['D24'])*1e4,'--',color=colors[119],label='$D(C_{T})_{24}$')
        D_all_ax.plot(group.Temperature,group.D*1e4,'ko',label='Experiment')
        D_all_ax.plot(sim_df['Temperature'],np.array(D_dict['Dz'])*1e4,'C9',label='Fit')
        
        P_fig = plt.figure()
        P_ax = P_fig.add_subplot(111)
        P_ax.plot(sim_df['Temperature'],P_dict['P3'],color=colors[499],LineWidth=2,label='$P_{3}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P6'],color=colors[199],LineWidth=2,label='$P_{6}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P12'],color=colors[159],LineWidth=2,label='$P_{12}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P24'],color=colors[119],LineWidth=2,label='$P_{24}$')
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
    
    pdf.savefig(D_all_fig)
    plt.close(D_all_fig)
    
    pdf.close()
    
### Read in data and set up for fitting
def main():
    data = pd.read_csv(sys.argv[1])

    ### Set up parameters and groups    
    data_WT = data[data['Sample'] == 'P2'] # DegP protease-dead, WT
    #data_WT = data_WT[data_WT['Concentration'] < 150]
    #data_WT = data_WT[data_WT['Temperature'] <= 30]
    #data_WT = data_WT[data_WT['Concentration'].isin([328.8, 258.3, 203, 159.5, 125.3, 98.5, 77.4, 60.8, 47.8, 37.5])]
    #data_WT = data_WT[data_WT['Concentration'].isin([862.7, 677.8, 532.6, 328.8, 159.5, 77.4, 37.5, 18.2, 8.8])]
    #data_WT.to_csv('DegP2_DLSparams_20190507.csv')
    #groups_WT = data_WT.groupby('Temperature')
    groups_WT = data_WT.groupby('Concentration')
    
    fit_params = Parameters()
    
    fit_params.add('dH6_0', value=-218000,vary=False)
    fit_params.add('K6_0', value=283000,vary=False)
    fit_params.add('dCp6', value=-2327.43)
    
    fit_params.add('dH12_0', value=445616.13)
    fit_params.add('K12_0', value=10481.6583)
    fit_params.add('dCp12', value=416.075)
    
    fit_params.add('dH24_0', value=104768.06)
    fit_params.add('K24_0', value=351.58)
    fit_params.add('dCp24', value=283.96)
    
    fit_params.add('Rh_3', value=4.856e-9)
    fit_params.add('n',value=-0.333)
    fit_params.add('kc3',value=-281,min=-1000,vary=False)
    fit_params.add('kc6',value=-200,min=-1000,vary=False)
    fit_params.add('kc12',value=-660.6,min=-1000,vary=False)
    fit_params.add('kc24',value=-674.18,min=-1000,vary=False)

    run_fit = 'y'
    
    if run_fit == 'y':
        # Fit data    
        result = minimize(objective, fit_params, method='leastsq', args=(1, groups_WT))
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

    #fit_params.add('dH6_0', value=-148125.831,vary=False)
    #fit_params.add('K6_0', value=756928.371,vary=False)
    #fit_params.add('dCp6', value=13606.9504)
    
    #fit_params.add('dH12_0', value=61009.3724)
    #fit_params.add('K12_0', value=402.618950)
    #fit_params.add('dCp12', value=2512.26815)
    
    #fit_params.add('dH24_0', value=1.4482e-11)
    #fit_params.add('K24_0', value=74291.9471)
    #fit_params.add('dCp24', value=2250.40047)
    
    #fit_params.add('Rh_3', value=4.5554e-9,vary=False)
    #fit_params.add('n',value=-0.333)
    #fit_params.add('kc3',value=-13294.2727)
    #fit_params.add('kc6',value=-11.2338337)
    #fit_params.add('kc12',value=-1860.37441)
    #fit_params.add('kc24',value=-245.855388)
    
    #############################################
    ###### GOOD PARAMS 2 with n =/= -0.333 ######
    #############################################
    #fit_params.add('dH6_0', value=-188572.9)
    #fit_params.add('K6_0', value=90909.09)
    #fit_params.add('dCp6', value=16808.2576)
    
    #fit_params.add('dH12_0', value=295095.147)
    #fit_params.add('K12_0', value=1272.90164)
    #fit_params.add('dCp12', value=-38999.0344)
    
    #fit_params.add('dH24_0', value=167902.760)
    #fit_params.add('K24_0', value=9919.21815)
    #fit_params.add('dCp24', value=-19202.2822)
    
    #fit_params.add('Rh_3', value=4.987e-9)
    #fit_params.add('n',value=-0.1801369,vary=False)
    #fit_params.add('kc3',value=-6195.78387)
    #fit_params.add('kc6',value=-212.285318)
    #fit_params.add('kc12',value=75.1876699)
    #fit_params.add('kc24',value=-879.288708)
    
    ### kd can't be negative params
    #fit_params.add('dH6_0', value=-148125.831,vary=False)
    #fit_params.add('K6_0', value=756928.371,vary=False)
    #fit_params.add('dCp6', value=-5000)
    
    #fit_params.add('dH12_0', value=200000)
    #fit_params.add('K12_0', value=5000)
    #fit_params.add('dCp12', value=-1000)
    
    #fit_params.add('dH24_0', value=300000)
    #fit_params.add('K24_0', value=74291.9471)
    #fit_params.add('dCp24', value=-6000)
    
    #fit_params.add('Rh_3', value=4.5554e-9)
    #fit_params.add('n',value=-0.333,vary=False)
    #fit_params.add('kc3',value=-100,min=-1000)
    #fit_params.add('kc6',value=-100,min=-1000)
    #fit_params.add('kc12',value=-600,min=-1000)
    #fit_params.add('kc24',value=-600,min=-1000)