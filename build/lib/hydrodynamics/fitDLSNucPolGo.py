#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 23:31:43 2020

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
    data = pd.read_csv(sys.argv[1])

    ### Set up parameters and groups    
    data_WT = data[data['Sample'] == 'P2'] # DegP protease-dead, WT
    data_WT = data[data['Temperature'] <= 50]
    data_WT = data_WT[data_WT['Concentration'].isin([862.7, 677.8, 532.6, 328.8, 159.5, 77.4, 37.5, 18.2, 8.8])]
    groups_WT = data_WT.groupby('Concentration')
    
    fit_params = Parameters()
    
    fit_params.add('dHn_0', value=-218000,vary=False)
    fit_params.add('Kn_0', value=283000,vary=False)
    fit_params.add('dCpn', value=14244.4370)
    
    fit_params.add('dHe_0', value=31194.0755)
    fit_params.add('Ke_0', value=34990.2191)
    fit_params.add('dCpe', value=-3117.01168)

    fit_params.add('T0', value=30+273.15,vary=False)
    
    fit_params.add('Rh_3', value=4.6965e-9, vary=True) # This is not currently being used due to AUC constraint
    fit_params.add('m',value=-0.333,vary=False) ### YOU HAVE CURRENTLY SET THE 3-6 SCALING TO BE -0.27!!!! ###
    fit_params.add('n',value=2,vary=False)
    fit_params.add('N',value=100,vary=False)
    #fit_params.add('k_d',value=-50)
    fit_params.add('kd1',value=-43.4085)  ### YOU CURRENTLY HAVE INDIVIDUAL NONIDEALITY CONSTANTS SET FOR 3/6/+ ###
    fit_params.add('kd2',value=-149.119274)
    fit_params.add('kd3',value=-607.217652)

    run_fit = 'n' ####### You currently have set the trimer D-value using AUC constraints!!!!!!!!!! #########
    
    if run_fit == 'y':
        # Fit data    
        result = minimize(objective, fit_params, method='nelder', args=(1, groups_WT))
        report_fit(result)
        opt_params = result.params
        plot_fit(groups_WT, opt_params)
        
        print('\n')
        fit_params.pretty_print(fmt='e',colwidth=12,columns=['value'])
        
    if run_fit == 'n':
        # Simulate data
        plot_fit(groups_WT, fit_params)
        fit_params.pretty_print(fmt='e',colwidth=12,columns=['value'])

### Nucleated polymerization
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
    
    #k_d = fit_params['k_d'].value
    kd1 = fit_params['kd1'].value
    kd2 = fit_params['kd2'].value
    kd3 = fit_params['kd3'].value
    
    n = fit_params['n'].value
    N = int(fit_params['N'].value)
    m = fit_params['m'].value
    
    # Set up dictionaries for species up to size N
    D_dict = {}
    c_dict = {}
    for x in range(1,N+1):
        D_dict[f"D{3*x}_0"] = np.array([])
        D_dict[f"D{3*x}"] = np.array([])
        c_dict[f"c{3*x}"] = np.array([])
    
    D_dict['Dz'] = np.array([])
    K_dict = {'Kn':np.array([]),'Ke':np.array([])}
    
    #start_time = timeit.default_timer()
    for y in range(len(Temperatures)):
            
        # Calculate equilibrium constants
        Kn = Kn_0*np.exp((dHn_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpn/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        Ke = Ke_0*np.exp((dHe_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpe/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        
        ################## Solve dimensionless trimer concentration
        constants = [(Concentrations[0]*Ke/3),Kn/Ke,n] # XT, sigma, nucleus size
        g_guess = 1e-5
        ffs_partial = partial(ffs3,constants)
        sol = opt.root(ffs_partial,g_guess,method='lm')
        c3 = sol.x[0]/Ke

        #Generate populations and concentrations from solver solutions, calculate diffusion coefficients
        D_num = 0
        D_numtemp = 0
        D_den = 0
        D_dentemp = 0
        D3_0 = StokesEinstein(Temperatures[y], fit_params['Rh_3'].value)
        for x in range(1,N+1):
            
            if x == 1:
                c3x = c3
                c_dict[f"c{3*x}"] = np.append(c_dict[f"c{3*x}"],c3x)
    
            if 1 < x <= n:
                c3x = Kn*c_dict[f"c{3*(x-1)}"][-1]*c3
                c_dict[f"c{3*x}"] = np.append(c_dict[f"c{3*x}"],c3x)
                
            elif x > n:
                c3x = Ke*c_dict[f"c{3*(x-1)}"][-1]*c3
                c_dict[f"c{3*x}"] = np.append(c_dict[f"c{3*x}"],c3x)
            
            # Calculate the numerator and denominator terms for average diffusion coefficient
            if x == 2:
                D3x_0 = D3_0*x**(-0.334)
            else:
                D3x_0 = D3_0*x**(m)
            
            ##### ALLOWING FOR 3-MER, 6-MER, AND >6-MER TO HAVE THEIR OWN NONIDEALITY COEFFICIENTS!!!!!!!! ########          
            if x == 1:
                D_numtemp = (x**2)*c3x*(D3x_0*(1 + kd1*Concentrations[0]))
                D_dict[f"D{3*x}"] = np.append(D_dict[f"D{3*x}"],D3x_0*(1 + kd1*Concentrations[0]))
                
            if x == 2:
                D_numtemp = (x**2)*c3x*(D3x_0*(1 + kd2*Concentrations[0]))
                D_dict[f"D{3*x}"] = np.append(D_dict[f"D{3*x}"],D3x_0*(1 + kd2*Concentrations[0]))
                
            elif 3 <= x <= N:
                D_numtemp = (x**2)*c3x*(D3x_0*(1 + kd3*Concentrations[0]))
                D_dict[f"D{3*x}"] = np.append(D_dict[f"D{3*x}"],D3x_0*(1 + kd3*Concentrations[0]))
            
            D_num = D_num + D_numtemp
            D_dentemp = (x**2)*c3x
            D_den = D_den + D_dentemp
            D_dict[f"D{3*x}_0"] = np.append(D_dict[f"D{3*x}_0"],D3x_0)
            #D_dict[f"D{3*x}"] = np.append(D_dict[f"D{3*x}"],D3x_0*(1 + k_d*Concentrations[0]))
        
        # Make dictionary of D values for fit, get average diffusion coefficient
        Dz = D_num/D_den
        D_dict['Dz'] = np.append(D_dict['Dz'],Dz)

        # Make dictionary of the K values
        K_dict['Kn'] = np.append(K_dict['Kn'],Kn)
        K_dict['Ke'] = np.append(K_dict['Ke'],Ke)
      
    return D_dict, c_dict, K_dict

### Nucleated polymerization solver - dimensionless
def ffs3(constants,g):
    
    XT, s, n = constants # Unpack constants
    
    first_term = (s**(-1))*((((s*g)**(n+1))*(n*s*g - n - 1) + s*g)/((s*g - 1)**2))
    second_term = -(s**(n - 1))*(((g**(n+1))*(n*g - n - 1))/((g - 1)**2))
    eq1 = -XT + (first_term + second_term)
    
    return eq1

### Calculate diffusion coefficients
def StokesEinstein(T, Rh):
    
    kB = 1.38065e-23
    #eta = (2.414e-5)*10**(247.8/(T-140)) # Water viscosity
    #Dt = (kB*T)/(6*np.pi*eta*Rh)
    eta_HSBuffer = (9.46568e-11)*np.power(T,4) - (1.20155e-7)*np.power(T,3) + (5.73768e-5)*np.square(T) - (1.22266e-2)*T + 9.82422e-1 # HS DLS Buffer Viscosity from SEDNTERP1
    Dt = (kB*T)/(6*np.pi*eta_HSBuffer*Rh)
    #Dt = (kB*T)/(6*np.pi*eta_HSBuffer*1.67*3.56e-9) # Assumed from AUC analysis: frictional ratio, s-value and R0 of the trimer S210A/Y444A mutant
    
    return Dt
    
### Minimization function for global fit
def objective(fit_params, x, data_WT):
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    resid = np.array([])
    for ind, group in data_WT:
        D_dict, c_dict, K_dict = supercageNucPol(fit_params, group)
        resid = np.append(resid,np.array(group.D.values*1e4) - D_dict['Dz']*1e4)
        
        #ax.plot(group.Temperature,group.D.values,'ko')
        #ax.plot(group.Temperature,D_dict['Dz'],'r')
        
    #ax.set_ylim(1.5e-11,8.5e-11)
    #plt.ion()
    #plt.pause(0.00001)
    #plt.ioff()
    #plt.close()
    
    #print("RSS: ",np.sum(np.square(resid)))
    #for k in fit_params.keys():
    #    print(fit_params[k])
    
    return resid
    
### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(data_WT, opt_params):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DLSglobalfits.pdf')
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = int(opt_params['N'].value/10) ### Only plot the first ten and the final size!
    
    D_fig = plt.figure()
    D_ax = D_fig.add_subplot(111)
    D_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    D_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    D_ax.yaxis.major.formatter._useMathText = True
    D_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
    D_ax.set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$',fontsize=14)
    D_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    D_ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    
    Cooperativity_fig = plt.figure()
    Cooperativity_ax = Cooperativity_fig.add_subplot(111)
    Cooperativity_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    Cooperativity_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    Cooperativity_ax.yaxis.major.formatter._useMathText = True
    Cooperativity_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
    Cooperativity_ax.set_ylabel('$log_{10}(\sigma)$',fontsize=14)
    Cooperativity_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    
    K_fig = plt.figure()
    K_ax = K_fig.add_subplot(111)
    K_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    K_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    K_ax.yaxis.major.formatter._useMathText = True
    K_ax.set_xlabel("1/T $K^{-1}$",fontsize=14)
    K_ax.set_ylabel('ln(K)',fontsize=14)
    K_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))

    for ind, group in data_WT:
        
        sim_df = pd.DataFrame()
        sim_df['Temperature'] = np.linspace(np.min(group.Temperature),np.max(group.Temperature),100)
        sim_df['Concentration'] = np.zeros(len(sim_df['Temperature']))
        sim_df['Concentration'] = group.Concentration.iloc[0]
        D_dict, c_dict, K_dict = supercageNucPol(opt_params, sim_df)

        D_ax.plot(group.Temperature,group.D*1e4,'ko',label='Experiment')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['Dz'])*1e4,'r',label='Fit')
        
        c_fig = plt.figure()
        c_ax = c_fig.add_subplot(111)
        c_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        c_ax.yaxis.major.formatter._useMathText = True
        c_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
        c_ax.set_ylabel('Concentration $\mu$M',fontsize=14)
        c_ax.set_title(f"DegP_2, $[M]_{{T}}$: {group.Concentration.iloc[0]} $\mu$M",fontsize=14)
        c_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        c_ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        
        for x in range(1,int(opt_params['N'].value)+1):
            if x == 1:
                D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}_0"])*1e4,label=f"$D_{3*x,0}$")
                D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}"])*1e4,'--',label=f"$D_{3*x}$")
                #print(f"D{3*x}_0: ",D_dict[f"D{3*x}_0"])
            if x == 2:
                D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}_0"])*1e4,label=f"$D_{3*x,0}$")
            
            elif x == opt_params['N'].value:
                D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}_0"])*1e4,label=f"$D_{3*x,0}$")
                D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}"])*1e4,'--',label=f"$D_{3*x}$")
                #c_ax.plot(sim_df['Temperature'],np.array(c_dict[f"c{3*x}"])*1e6,LineWidth=2,label=f"$c_{3*x}$")
                print(f"### The concentration of the {3*opt_params['N'].value}-mer at PT = {group.Concentration.iloc[0]} uM as a function of temperature is: ####")
                print(c_dict[f"c{3*x}"])
                print('\n')
                
            c_ax.plot(sim_df['Temperature'],np.array(c_dict[f"c{3*x}"])*1e6,LineWidth=2,label=f"$c_{3*x}$")
            Cooperativity_ax.plot(sim_df['Temperature'],np.log10(K_dict['Kn']/K_dict['Ke']),LineWidth=2,color='#fc8d59')
            K_ax.plot(1./(sim_df['Temperature'] + 273.15),np.log(K_dict['Kn']),LineWidth=2,color='k')
            K_ax.plot(1./(sim_df['Temperature'] + 273.15),np.log(K_dict['Ke']),LineWidth=2,color='r')
            
        #c_ax.legend(loc='upper left',fontsize=12,frameon=False)
        pdf.savefig(c_fig)
        plt.close(c_fig)
    
    pdf.savefig(Cooperativity_fig)
    plt.close(Cooperativity_fig)
    pdf.savefig(D_fig)
    plt.close(D_fig)
    pdf.savefig(K_fig)
    plt.close(K_fig)
    pdf.close()

R = 1.9872e-3 # Gas constant for calculating affinities
R = R*4.184*1000 # Thermodynamic parameters have to be in J/mol and J/mol*K to match this gas constant
main()

    #fit_params.add('dHn_0', value=-232372.011,vary=False)
    #fit_params.add('Kn_0', value=278467.148,vary=False)
    #fit_params.add('dCpn', value=0,vary=False)
    
    #fit_params.add('dHe_0', value=114435.568)
    #fit_params.add('Ke_0', value=300000)
    #fit_params.add('dCpe', value=-11313.9294)

    #fit_params.add('T0', value=30+273.15, vary=False)
    
    #fit_params.add('Rh_3', value=4.9e-9)
    #fit_params.add('m',value=-1/3)
    #fit_params.add('n',value=2,vary=False)
    #fit_params.add('N',value=300,vary=False)
    #fit_params.add('k_d',value=5.29,vary=False)
    
#    fit_params.add('dHn_0', value=-232372.011,vary=False)
#    fit_params.add('Kn_0', value=278467.148,vary=False)
#    fit_params.add('dCpn', value=0,vary=False)
#    
#    fit_params.add('dHe_0', value=-13355.2890)
#    fit_params.add('Ke_0', value=70420.0250)
#    fit_params.add('dCpe', value=-4018.61821)
#
#    fit_params.add('T0', value=30+273.15, vary=False)
#    
#    fit_params.add('Rh_3', value=4.9423e-9)
#    fit_params.add('m',value=-0.35966112)
#    fit_params.add('n',value=2,vary=False)
#    fit_params.add('N',value=300,vary=False)
#    fit_params.add('k_d',value=5.29,vary=False)
    
# Good fit but not physically realistic parameters
    
#    fit_params.add('dHn_0', value=54145.0340)
#    fit_params.add('Kn_0', value=1268.58303)
#    fit_params.add('dCpn', value=274.517372)
    
#    fit_params.add('dHe_0', value=111668.056)
#    fit_params.add('Ke_0', value=1554.65628)
#    fit_params.add('dCpe', value=-9559.477)

#    fit_params.add('T0', value=30+273.15, vary=False)
#    
#    fit_params.add('Rh_3', value=5.5865e-9)
#    fit_params.add('m',value=-1.70918776)
#    fit_params.add('n',value=2,vary=False)
#    fit_params.add('N',value=100,vary=False)
#    fit_params.add('k_d',value=1.98597364)
    
        #fit_params.add('dHn_0', value=-150000)
    #fit_params.add('Kn_0', value=150000)
    #fit_params.add('dCpn', value=0)
    
    #fit_params.add('dHe_0', value=20000)
    #fit_params.add('Ke_0', value=80000)
    #fit_params.add('dCpe', value=-500)

    #fit_params.add('T0', value=30+273.15,vary=False)
    
    #fit_params.add('Rh_3', value=5.1e-9,vary=False)
    #fit_params.add('m',value=-0.3333,vary=False) ### YOU HAVE CURRENTLY SET THE 3-6 SCALING TO BE -0.3333!!!! ###
    #fit_params.add('n',value=2,vary=False)
    #fit_params.add('N',value=300,vary=False)
    #fit_params.add('k_d',value=0)
    
        #fit_params.add('dHe_0', value=58264.6440)
    #fit_params.add('Ke_0', value=21615.0952)
    #fit_params.add('dCpe', value=-5676.97481)