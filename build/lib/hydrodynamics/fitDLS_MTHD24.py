#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 07:07:30 2019

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

def MTHD24(fit_params, group_WT):
    
    # Set up arrays for calculating concentrations as a function of temperature and total monomer concentration
    Temperatures = np.array(group_WT.Temperature + 273.15)
    Concentrations = np.array(group_WT.Concentration*1e-6)
    
    K3 = np.zeros(len(Temperatures))
    K6 = np.zeros(len(Temperatures))
    K12 = np.zeros(len(Concentrations))
    K24 = np.zeros(len(Concentrations))
    
    M1 = np.zeros(len(Concentrations))
    M3 = np.zeros(len(Concentrations))
    M6 = np.zeros(len(Concentrations))
    M12 = np.zeros(len(Concentrations))
    M24 = np.zeros(len(Concentrations))
    
    M1_num = np.zeros(len(Concentrations))
    M3_num = np.zeros(len(Concentrations))
    M6_num = np.zeros(len(Concentrations))
    M12_num = np.zeros(len(Concentrations))
    M24_num = np.zeros(len(Concentrations))
    
    M1_den = np.zeros(len(Concentrations))
    M3_den = np.zeros(len(Concentrations))
    M6_den = np.zeros(len(Concentrations))
    M12_den = np.zeros(len(Concentrations))
    M24_den = np.zeros(len(Concentrations))
    
    D1_0 = np.zeros(len(Concentrations))
    D3_0 = np.zeros(len(Concentrations))
    D6_0 = np.zeros(len(Concentrations))
    D12_0 = np.zeros(len(Concentrations))
    D24_0 = np.zeros(len(Concentrations))
    
    Dz_MTHD24 = np.zeros(len(Concentrations))
    
    P_dict = {'P1':[],'P3':[],'P6':[],'P12':[],'P24':[]}
    D_dict = {'D1_0':[],'D3_0':[],'D6_0':[],'D12_0':[],'D24_0':[]}
    K_dict = {'K3':[],'K6':[],'K12':[],'K24':[]}
    
    dH3_0 = fit_params['dH3_0'].value
    K3_0= abs(fit_params['K3_0'].value)
    dCp3 = fit_params['dCp3'].value
    dH6_0 = fit_params['dH6_0'].value
    K6_0 = abs(fit_params['K6_0'].value)
    dCp6 = fit_params['dCp6'].value
    dH12_0 = fit_params['dH12_0'].value
    K12_0 = abs(fit_params['K12_0'].value)
    dCp12 = fit_params['dCp12'].value
    dH24_0 = fit_params['dH24_0'].value
    K24_0 = abs(fit_params['K24_0'].value)
    dCp24 = fit_params['dCp24'].value
    k_c = -abs(fit_params['k_c'].value)

    for y in range(len(Temperatures)):
            
        # Calculate equilibrium constants
        K3[y] = K3_0*np.exp((dH3_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCp3/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        K6[y] = K6_0*np.exp((dH6_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCp6/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        K12[y] = K12_0*np.exp((dH12_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCp12/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        K24[y] = K24_0*np.exp((dH24_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCp24/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        
        # Get concentrations from solver
        solutions = []
        p = [Concentrations[0],1e-6,1e-6,1e-6,1e-6]
        q = [Concentrations[0],K3[y],K6[y],K12[y],K24[y]]
        ffs_partial = partial(ffs,q)
        solutions.append(opt.root(ffs_partial,p,method='lm',tol=1e-10))
        
        for sol in solutions:
            M1[y] = sol.x[0]
            M3[y] = sol.x[1]
            M6[y] = sol.x[2]
            M12[y] = sol.x[3]
            M24[y] = sol.x[4]
        
        #iter_conc = Concentrations[0]
        #P_sum = (M1[y] + 3*M3[y] + 6*M6[y] +12*M12[y] + 24*M24[y])/Concentrations[0]
        #print(f"Populations sum to: {P_sum}\nThe temperature is {Temperatures[y]-273.15}\n")
#        while P_sum > 1.05 or P_sum < 0.95:
#            solutions = []
#            iter_conc = iter_conc*0.95
#            p = [iter_conc,1e-6,1e-6,1e-6,1e-6]
#            q = [Concentrations[0],K3[y],K6[y],K12[y],K24[y]]
#            ffs_partial = partial(ffs,q)
#            solutions.append(opt.root(ffs_partial,p,method='broyden1'))
#        
#            for sol in solutions:
#                M1[y] = sol.x[0]
#                M3[y] = sol.x[1]
#                M6[y] = sol.x[2]
#                M12[y] = sol.x[3]
#                M24[y] = sol.x[4]
        
        #if np.round((M1[y] + 3*M3[y] + 6*M6[y] + 12*M12[y] + 24*M24[y]),12) > Concentrations[0]:
        #    print("Populations don't sum to 1!\n")
        #    print(f"{M1[y] + 3*M3[y] + 6*M6[y] + 12*M12[y] + 24*M24[y]} > {Concentrations[0]}")
        
        # Calculate the numerator and denominator terms and finally the average diffusion coefficient
        D1_0[y] = StokesEinstein(Temperatures[y], fit_params['Rh_1'].value)
        D3_0[y] = StokesEinstein(Temperatures[y], fit_params['Rh_3'].value)
        D6_0[y] = StokesEinstein(Temperatures[y], fit_params['Rh_6'].value)
        D12_0[y] = StokesEinstein(Temperatures[y], fit_params['Rh_12'].value)
        D24_0[y] = StokesEinstein(Temperatures[y], fit_params['Rh_24'].value)
            
        M1_num[y] = 1*M1[y]*(D1_0[y]*(1+ k_c*M1[y]))
        M3_num[y] = 9*M3[y]*(D3_0[y]*(1 + k_c*M3[y]))
        M6_num[y] = 36*M6[y]*(D6_0[y]*(1 + k_c*M6[y]))
        M12_num[y] = 144*M12[y]*(D12_0[y]*(1 + k_c*M12[y]))
        M24_num[y] = 576*M24[y]*(D24_0[y]*(1 + k_c*M24[y]))
            
        M1_den[y] = 1*M1[y]
        M3_den[y] = 9*M3[y]
        M6_den[y] = 36*M6[y]
        M12_den[y] = 144*M12[y]
        M24_den[y] = 576*M24[y]
            
        Dz_MTHD24[y] = (M1_num[y] + M3_num[y] + M6_num[y] + M12_num[y] + M24_num[y])/(M1_den[y] + M3_den[y] + M6_den[y] + M12_den[y] + M24_den[y])
            
        #print(f"Temperature:{Temperatures[x]}, Concentration:{Concentrations[x]}")
        
        # Make dictionary of the populations
        P_dict['P1'].append(1*M1[y]/Concentrations[y])
        P_dict['P3'].append(3*M3[y]/Concentrations[y])
        P_dict['P6'].append(6*M6[y]/Concentrations[y])
        P_dict['P12'].append(12*M12[y]/Concentrations[y])
        P_dict['P24'].append(24*M24[y]/Concentrations[y])
        
        # Make dictionary of the D0 values
        D_dict['D1_0'].append(D1_0[y])
        D_dict['D3_0'].append(D3_0[y])
        D_dict['D6_0'].append(D6_0[y])
        D_dict['D12_0'].append(D12_0[y])
        D_dict['D24_0'].append(D24_0[y])
        
        # Make dictionary of the K values
        K_dict['K3'].append(K3[y])
        K_dict['K6'].append(K6[y])
        K_dict['K12'].append(K12[y])
        K_dict['K24'].append(K24[y])

        #print(f"### WT ### T: {Temperatures[x]}, M: {M[x]}, Tr: {Tr[x]}, H: {H[x]}, Dd: {Dd[x]}, Dz_MTHD: {Dz_MTHD[x]}")
        #print(f"### WT ###\nK_H: {K_H}\nK_Dd: {K_Dd}\nPsum: {(3*Tr[x] + 6*H[x] + 12*Dd[x])/Concentrations[x]}\nTemperature: {Temperatures[x]}\n")
        #print(P_dict,'Concentrations:',Concentrations[x])
            
    return Dz_MTHD24, P_dict, K_dict, D_dict

# Solver function for getting concentrations of oligomers
def ffs(q,p):
    
    CT, K3, K6, K12, K24 = q # Unpack constants
    M1, M3, M6, M12, M24 = p # Unpack variables
    
    eq1 = -CT + M1 + 3*M3 + 6*M6 + 12*M12 + 24*M24
    eq2 = K3*M1**3 - M3
    eq3 = K6*M3**2 - M6
    eq4 = K12*M6**2 - M12
    eq5 = K24*M12**2 - M24
    
    return [eq1, eq2, eq3, eq4, eq5]

### Calculate diffusion coefficients
def StokesEinstein(T, Rh):
    
    kB = 1.38065e-23
    eta = (2.414e-5)*10**(247.8/(T-140))
    Dt = (kB*T)/(6*np.pi*eta*Rh)
    
    return Dt
    
### Minimization function for global fit
def objective(fit_params, x, data_WT):
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    resid = []
    for ind, group in data_WT:
        #resid = group.D.values*0.0
        Dz_MTHD24,P_dict,K_dict,D_dict = MTHD24(fit_params, group)
        resid.append(group.D.values*1e4 - Dz_MTHD24*1e4)
        
        #ax.plot(group.Temperature,group.D.values,'ko')
        #ax.plot(group.Temperature,Dz_MTHD24,'r')
     
    #print(fit_params['K12_0'].value)
    #print(fit_params['K24_0'].value)
    
    resid = np.ravel(resid)
    #print(resid,'\n')
    #plt.ion()
    #plt.pause(0.00001)
    #plt.ioff()
    #input()
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
        Dz_MTHD24, P_dict, K_dict, D_dict = MTHD24(opt_params, sim_df)
        
        D_fig = plt.figure()
        D_ax = D_fig.add_subplot(111)
        #axs[0].plot(group.Concentration,group.D*1e4,'ko')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D1_0'])*1e4)
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D3_0'])*1e4)
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D6_0'])*1e4)
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D12_0'])*1e4)
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['D24_0'])*1e4)
        D_ax.plot(group.Temperature,group.D*1e4,'ko')
        D_ax.plot(sim_df['Temperature'],Dz_MTHD24*1e4,'r',LineWidth=2)
        D_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        D_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        D_ax.yaxis.major.formatter._useMathText = True
        D_ax.set_title(f"DegP_2, $[M]_{{T}}$: {group.Concentration.iloc[0]} $\mu$M",fontsize=14)
        D_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
        D_ax.set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$',fontsize=14)
        D_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        
        P_fig = plt.figure()
        P_ax = P_fig.add_subplot(111)
        P_ax.plot(sim_df['Temperature'],P_dict['P1'],'k',LineWidth=2,label='$P_{1}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P3'],color=colors[499],LineWidth=2,label='$P_{3}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P6'],color=colors[150],LineWidth=2,label='$P_{6}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P12'],color=colors[80],LineWidth=2,label='$P_{12}$')
        P_ax.plot(sim_df['Temperature'],P_dict['P24'],color=colors[20],LineWidth=2,label='$P_{24}$')
        P_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        P_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        P_ax.yaxis.major.formatter._useMathText = True
        P_ax.set_title(f"DegP_2, $[M]_{{T}}$: {group.Concentration.iloc[0]} $\mu$M",fontsize=14)
        P_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
        P_ax.set_ylabel('Population',fontsize=14)
        P_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        P_ax.legend(loc='upper left',fontsize=12,frameon=False)
        
        pdf.savefig(D_fig)
        pdf.savefig(P_fig)
        plt.close(D_fig)
        plt.close(P_fig)
        
    pdf.close()
    
### Read in data and set up for fitting
def main():
    data = pd.read_csv(sys.argv[1])

    ### Set up parameters and groups    
    data_WT = data[data['Sample'] == 'P2'] # DegP protease-dead, WT
    #data_WT.to_csv('DegP2_DLSparams_20190507.csv')
    #groups_WT = data_WT.groupby('Temperature')
    groups_WT = data_WT.groupby('Concentration')
    
    fit_params = Parameters()
    fit_params.add('dH3_0', value=-300000)
    fit_params.add('K3_0', value=1e16)
    fit_params.add('dCp3', value=10000)
    fit_params.add('dH6_0', value=-200000)
    fit_params.add('K6_0', value=1e11)
    fit_params.add('dCp6', value=-3000)
    fit_params.add('dH12_0', value=280000)
    fit_params.add('K12_0', value=24.32)
    fit_params.add('dCp12', value=-10000)
    fit_params.add('dH24_0', value=580000)
    fit_params.add('K24_0', value=0.0544)
    fit_params.add('dCp24', value=-12910)
    fit_params.add('T0', value=5+273.15, vary=False)

    fit_params.add('Rh_1', value=3.1e-9, vary=False)
    fit_params.add('Rh_3', value=4.7e-9, vary=False)
    fit_params.add('Rh_6', value=5.8e-9, vary=False)
    fit_params.add('Rh_12', value=8.8e-9, vary=False)
    fit_params.add('Rh_24', value=20e-9, vary=False)
    fit_params.add('k_c', value=-10000)

    run_fit = 'n'
    
    if run_fit == 'y':
        # Fit data    
        result = minimize(objective, fit_params, method='lm', args=(1, groups_WT))
        report_fit(result)
        opt_params = result.params
        plot_fit(groups_WT, opt_params)
    
    if run_fit == 'n':
        # Simulate data
        plot_fit(groups_WT, fit_params)

R = 1.9872e-3 # Gas constant for calculating affinities
R = R*4.184*1000 # Thermodynamic parameters have to be in J/mol and J/mol*K to match this gas constant
main()



