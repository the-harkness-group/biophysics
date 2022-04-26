#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:49:39 2020

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
    data_WT = data[data['Sample'] == 'P2'] # DegP_2 S210A, protease-dead
    #data_WT = data[data['Temperature'] <= 50]
    #data_WT = data_WT[data_WT['Concentration'].isin([862.7, 677.8, 532.6, 328.8, 159.5, 77.4, 37.5, 18.2, 8.8])]
    data_WT = data_WT[data_WT['Concentration'].isin([862.7, 532.6, 328.8, 159.5, 77.4, 18.2, 8.8])]
    groups_WT = data_WT.groupby('Concentration')
    
    #data_WT = data[data.Sample == 'P3'] # DegP_2 S210A, protease-dead
    #data_WT = data_WT[data_WT['Temperature'] <= 50]
    #data_WT = data_WT[data_WT['Concentration'].isin([862.7, 677.8, 532.6, 328.8, 159.5, 77.4, 37.5, 18.2, 8.8])]
    #data_WT = data_WT[data_WT['Concentration'].isin([514.8, 128.7, 42.9, 12.9])]
    #data_WT = data_WT[data_WT['Concentration'].isin([862.7, 532.6, 328.8, 98.5, 37.5, 8.8])]
    #groups_WT = data_WT.groupby('Concentration')
    
    #print(data[data.Sample == 'P2'])
    
    #data_WT = data[data['Sample'] == 'P7'] # DegP_7 S210A/F289A, trimer mutant
    #data_WT = data[data['Temperature'] <= 50]
    #data_WT = data_WT[data_WT['Concentration'].isin([318.2, 221.3, 153.9, 107.1, 74.5, 51.8, 36.0, 25.1, 17.4, 10.1])]
    #groups_WT = data_WT.groupby('Concentration')
    
    #data_WT = data[data['Sample'] == 'P8'] # DegP_8 S210A/R325A, signalling loop "hexamer" mutant"
    #data_WT = data[data['Temperature'] <= 50]
    #data_WT = data_WT[data_WT['Concentration'].isin([558.1, 365.3, 239.1, 156.5, 102.4, 67.0, 43.9, 28.7, 18.8, 10.0])]
    #groups_WT = data_WT.groupby('Concentration')
    
    fit_params = Parameters()
    
    fit_params.add('dHn1_0', value=-147722.877,vary=True) # DegP_2 S210A REFOLDED PARAMETERS, LOW SALT BUFFER 25 mM NaH2PO4 1 mM EDTA, WATER VISCOSITY
    fit_params.add('Kn1_0', value=236880.853,vary=True)
    #fit_params.add('dCpn1', value=-7508.0233,vary=True)
    fit_params.add('dCpn1', value=0,vary=False)
    
    #fit_params.add('dHn1_0', value=-87006.2126,vary=True) # DegP_2 S210A REFOLDED PARAMETERS, LOW SALT BUFFER 25 mM NaH2PO4 1 mM EDTA, WATER VISCOSITY
    #fit_params.add('Kn1_0', value=1825765.28,vary=True)
    #fit_params.add('dCpn1', value=-13895.6561,vary=True)
    
    #fit_params.add('dHn1_0', value=-50000,vary=True) # DegP_2 S210A REFOLDED PARAMETERS, LOW SALT BUFFER 25 mM NaH2PO4 1 mM EDTA, WATER VISCOSITY
    #fit_params.add('Kn1_0', value=100,vary=True)
    #fit_params.add('dCpn1', value=0,vary=True)
    
    fit_params.add('dHn2_0', value=-19107.7095,vary=True)
    fit_params.add('Kn2_0', value=73865.3611,vary=True)
    #fit_params.add('dCpn2', value=-1135.60117)
    fit_params.add('dCpn2', value=0,vary=False)
    
    #fit_params.add('dHn2_0', value=-30690.0512,vary=True)
    #fit_params.add('Kn2_0', value=202024.773,vary=True)
    #fit_params.add('dCpn2', value=-6665.24459)
    
    #fit_params.add('dHn2_0', value=-5000,vary=True)
    #fit_params.add('Kn2_0', value=3e6,vary=True)
    #fit_params.add('dCpn2', value=-500)
    
    fit_params.add('dHe_0', value=7235.25468)
    fit_params.add('Ke_0', value=80442.8321)
    #fit_params.add('dCpe', value=-2598.34085)
    fit_params.add('dCpe', value=0,vary=False)
    
    #fit_params.add('dHe_0', value=105299.804)
    #fit_params.add('Ke_0', value=2618.01750)
    #fit_params.add('dCpe', value=1027.30935)
    
    #fit_params.add('dHe_0', value=10000)
    #fit_params.add('Ke_0', value=8e5)
    #fit_params.add('dCpe', value=-500)

    fit_params.add('T0', value=30+273.15,vary=False)
    
    fit_params.add('Rh_3', value=5.0e-9, vary=False,min=4e-9,max=6e-9)
    fit_params.add('h_scale',value=-0.333,vary=False) # >6-mer Diffusion Constant Scaling
    fit_params.add('j_scale',value=-0.15,vary=False) # 3-6-mer Diffusion Constant Scaling
    fit_params.add('n',value=12,vary=False)
    #fit_params.add('n',value=2,vary=False)
    fit_params.add('N',value=100,vary=False)
    fit_params.add('k_d',value=0,vary=False)

    run_fit = 'y' # YOU HAVE BUFFER VISCOSITY SET TO HIGH SALT BUFFER IN D2O!!!
    plot_minimization = 1 # 1 = yes, 0 = no
    
    if run_fit == 'y':
        
        # Fit data    
        result = minimize(objective, fit_params, method='nelder', args=(plot_minimization, groups_WT))
        report_fit(result)
        opt_params = result.params
        plot_fit(groups_WT, opt_params)
        
        print('\n')
        opt_params.pretty_print(fmt='e',colwidth=12,columns=['value']) # Print nice paramater table
        
    if run_fit == 'n':
        
        # Simulate data
        plot_fit(groups_WT, fit_params)
        fit_params.pretty_print(fmt='e',colwidth=12,columns=['value'])


### Nucleated polymerization
def supercageNucPol(fit_params, group_WT):
    
    Temperatures = np.array(group_WT.Temperature + 273.15)
    Concentrations = np.array(group_WT.Concentration*1e-6)
    
    # Set up thermodynamic and hydrodynamic parameters
    dHn1_0 = fit_params['dHn1_0'].value
    Kn1_0 = abs(fit_params['Kn1_0'].value)
    dCpn1 = fit_params['dCpn1'].value
    
    dHn2_0 = fit_params['dHn2_0'].value
    Kn2_0 = abs(fit_params['Kn2_0'].value)
    dCpn2 = fit_params['dCpn2'].value
    
    dHe_0 = fit_params['dHe_0'].value
    Ke_0 = abs(fit_params['Ke_0'].value)
    dCpe = fit_params['dCpe'].value
    
    k_d = fit_params['k_d'].value
    
    n = fit_params['n'].value
    N = int(fit_params['N'].value)
    h_scale = fit_params['h_scale'].value
    j_scale = fit_params['j_scale'].value
    
    # Set up dictionaries for species up to size N
    D_dict = {}
    c_dict = {}
    P_dict = {}
    for x in range(1,N+1):
        
        if x == 2:
            
            D_dict[f"D1{3*x}_0"] = np.array([])
            D_dict[f"D1{3*x}"] = np.array([])
            c_dict[f"c1{3*x}"] = np.array([])
            P_dict[f"P1{3*x}"] = np.array([])
            
            D_dict[f"D2{3*x}_0"] = np.array([])
            D_dict[f"D2{3*x}"] = np.array([])
            c_dict[f"c2{3*x}"] = np.array([])
            P_dict[f"P2{3*x}"] = np.array([])
            
        else:
            
            D_dict[f"D{3*x}_0"] = np.array([])
            D_dict[f"D{3*x}"] = np.array([])
            c_dict[f"c{3*x}"] = np.array([])
            P_dict[f"P{3*x}"] = np.array([])
            
    D_dict['Dz'] = np.array([])
    K_dict = {'Kn1':np.array([]),'Kn2':np.array([]),'Ke':np.array([])}
    
    #start_time = timeit.default_timer()
    X_guess = 1e-5
    for y in range(len(Temperatures)):
            
        # Calculate equilibrium constants
        Kn1 = Kn1_0*np.exp((dHn1_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpn1/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        Kn2 = Kn2_0*np.exp((dHn2_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpn2/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        Ke = Ke_0*np.exp((dHe_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpe/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        
        ################## Solve dimensionless trimer concentration
        constants = [(Concentrations[0]*Ke/3),Kn1/Kn2,Kn2/Ke,n] # XT, alpha1 = Kn1/Kn2, alpha2 = Kn2/Ke, nucleus size
        #X_guess = 1e-5
        ffs_partial = partial(ffs3,constants)
        sol = opt.root(ffs_partial,X_guess,method='lm')
        c3 = sol.x[0]/Ke
        
        # Calculate average oligomer size for N+1 to infinity
        #x3 = c3*Ke # dimensionless trimer concentration
        #ff = np.power((Kn2/Ke),(n-1))*(np.power(x3,N+1)/(1 - x3)) # dimensionless fiber concentration N+1 to infinity
        #zf = np.power(Kn2/Ke,n-1)*(-(np.power(x3,N+1)*(N*x3-N-1))/np.square(x3-1)) # dimensionless Z concentration N+1 to infinity
        #fl = zf/ff # average fiber length for N+1 to infinity is Z/F
        #print(f"The average fiber length above size N is : {fl}")

        #Generate populations and concentrations from solver solutions, calculate diffusion coefficients
        D_num = 0
        D_numtemp = 0
        D_den = 0
        D_dentemp = 0
        D3_0 = StokesEinstein(Temperatures[y], fit_params['Rh_3'].value)
        for x in range(1,N+1):
            
            D3x_0 = D3_0*x**(h_scale)
            
            if x == 1:
                c3x = c3
                c_dict[f"c{3*x}"] = np.append(c_dict[f"c{3*x}"],c3x) # Trimer
                P_dict[f"P{3*x}"] = np.append(P_dict[f"P{3*x}"],3*x*c3x/Concentrations[0])
                
                D_numtemp = (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                D_num = D_num + D_numtemp
                D_dentemp = (x**2)*c3x
                D_den = D_den + D_dentemp
                
                D_dict[f"D{3*x}_0"] = np.append(D_dict[f"D{3*x}_0"],D3x_0)
                D_dict[f"D{3*x}"] = np.append(D_dict[f"D{3*x}"],D3x_0*(1 + k_d*Concentrations[0]))
                
            if 1 < x <= n:
                
                if x == 2:
                
                    # Hexamer 1
                    D3x_0 = D3_0*x**(j_scale)
                    c3x = Kn1*c_dict[f"c{3*(x-1)}"][-1]*c3
                    c_dict[f"c1{3*x}"] = np.append(c_dict[f"c1{3*x}"],c3x) # Hexamer 1 dictated by trimer and Kn1
                    P_dict[f"P1{3*x}"] = np.append(P_dict[f"P1{3*x}"],3*x*c3x/Concentrations[0]) # Hexamer 1 dictated by trimer and Kn1
                
                    D_numtemp = (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                    D_num = D_num + D_numtemp
                    D_dentemp = (x**2)*c3x
                    D_den = D_den + D_dentemp
                
                    D_dict[f"D1{3*x}_0"] = np.append(D_dict[f"D1{3*x}_0"],D3x_0)
                    D_dict[f"D1{3*x}"] = np.append(D_dict[f"D1{3*x}"],D3x_0*(1 + k_d*Concentrations[0]))
                
                    # Hexamer 2
                    D3x_0 = D3_0*x**(j_scale)
                    c3x = Kn2*c_dict[f"c{3*(x-1)}"][-1]*c3
                    c_dict[f"c2{3*x}"] = np.append(c_dict[f"c2{3*x}"],c3x) # Hexamer 2 dictated by trimer and Kn2
                    P_dict[f"P2{3*x}"] = np.append(P_dict[f"P2{3*x}"],3*x*c3x/Concentrations[0]) # Hexamer 2 dictated by trimer and Kn2
                
                    D_numtemp = (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                    D_num = D_num + D_numtemp
                    D_dentemp = (x**2)*c3x
                    D_den = D_den + D_dentemp
                
                    D_dict[f"D2{3*x}_0"] = np.append(D_dict[f"D2{3*x}_0"],D3x_0)
                    D_dict[f"D2{3*x}"] = np.append(D_dict[f"D2{3*x}"],D3x_0*(1 + k_d*Concentrations[0]))
                
                if x == 3:
                    c3x = Kn2*c_dict[f"c2{3*(x-1)}"][-1]*c3
                    c_dict[f"c{3*x}"] = np.append(c_dict[f"c{3*x}"],c3x) # 9-mer dictated by hexamer 2 and Kn
                    P_dict[f"P{3*x}"] = np.append(P_dict[f"P{3*x}"],3*x*c3x/Concentrations[0])
                
                    D_numtemp = (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                    D_num = D_num + D_numtemp
                    D_dentemp = (x**2)*c3x
                    D_den = D_den + D_dentemp
                
                    D_dict[f"D{3*x}_0"] = np.append(D_dict[f"D{3*x}_0"],D3x_0)
                    D_dict[f"D{3*x}"] = np.append(D_dict[f"D{3*x}"],D3x_0*(1 + k_d*Concentrations[0]))
                
                if x > 3:
                    c3x = Kn2*c_dict[f"c{3*(x-1)}"][-1]*c3
                    c_dict[f"c{3*x}"] = np.append(c_dict[f"c{3*x}"],c3x) # >9-mer dictated by 9-mer and Kn
                    P_dict[f"P{3*x}"] = np.append(P_dict[f"P{3*x}"],3*x*c3x/Concentrations[0])
                
                    D_numtemp = (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                    D_num = D_num + D_numtemp
                    D_dentemp = (x**2)*c3x
                    D_den = D_den + D_dentemp
                
                    D_dict[f"D{3*x}_0"] = np.append(D_dict[f"D{3*x}_0"],D3x_0)
                    D_dict[f"D{3*x}"] = np.append(D_dict[f"D{3*x}"],D3x_0*(1 + k_d*Concentrations[0]))
                    
            if x > n:
                
                if x == 3:
                    c3x = Ke*c_dict[f"c2{3*(x-1)}"][-1]*c3
                    c_dict[f"c{3*x}"] = np.append(c_dict[f"c{3*x}"],c3x) # 9-mer dictated by hexamer 2 and Ke
                    P_dict[f"P{3*x}"] = np.append(P_dict[f"P{3*x}"],3*x*c3x/Concentrations[0])
                    
                    D_numtemp = (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                    D_num = D_num + D_numtemp
                    D_dentemp = (x**2)*c3x
                    D_den = D_den + D_dentemp
                
                    D_dict[f"D{3*x}_0"] = np.append(D_dict[f"D{3*x}_0"],D3x_0)
                    D_dict[f"D{3*x}"] = np.append(D_dict[f"D{3*x}"],D3x_0*(1 + k_d*Concentrations[0]))
                
                if x > 3:
                    c3x = Ke*c_dict[f"c{3*(x-1)}"][-1]*c3
                    c_dict[f"c{3*x}"] = np.append(c_dict[f"c{3*x}"],c3x) # >9-mer dictated by 9-mer and Ke
                    P_dict[f"P{3*x}"] = np.append(P_dict[f"P{3*x}"],3*x*c3x/Concentrations[0])
                
                    D_numtemp = (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                    D_num = D_num + D_numtemp
                    D_dentemp = (x**2)*c3x
                    D_den = D_den + D_dentemp
                
                    D_dict[f"D{3*x}_0"] = np.append(D_dict[f"D{3*x}_0"],D3x_0)
                    D_dict[f"D{3*x}"] = np.append(D_dict[f"D{3*x}"],D3x_0*(1 + k_d*Concentrations[0]))
        
        # Make dictionary of D values for fit, get average diffusion coefficient
        Dz = D_num/D_den
        D_dict['Dz'] = np.append(D_dict['Dz'],Dz)

        # Make dictionary of the K values
        K_dict['Kn1'] = np.append(K_dict['Kn1'],Kn1)
        K_dict['Kn2'] = np.append(K_dict['Kn2'],Kn2)
        K_dict['Ke'] = np.append(K_dict['Ke'],Ke)
        
        X_guess = c3*Ke # Solver guess for next temperature is solution at the i-1 temperature for stability
      
    return D_dict, c_dict, K_dict, P_dict


### Nucleated polymerization solver - dimensionless
def ffs3(constants,X):
    
    XT, a1, a2, n = constants # Unpack constants
    
    first_term = (a2**(-1))*((((a2*X)**(n+1))*(n*a2*X - n - 1) + a2*X)/((a2*X - 1)**2))
    second_term = -(a2**(n - 1))*(((X**(n+1))*(n*X - n - 1))/((X - 1)**2))
    eq1 = -XT + 2.*a1*a2*np.square(X) + first_term + second_term
    
    return eq1


### Calculate diffusion coefficients
def StokesEinstein(T, Rh):
    
    kB = 1.38065e-23
    #eta = (2.414e-5)*10**(247.8/(T-140)) # Water viscosity
    #Dt = (kB*T)/(6*np.pi*eta*Rh)
    eta_HSBuffer = (9.46568e-11)*np.power(T,4) - (1.20155e-7)*np.power(T,3) + (5.73768e-5)*np.square(T) - (1.22266e-2)*T + 9.82422e-1 # HS DLS Buffer Viscosity from SEDNTERP1
    #eta_HSBuffer_D2O = 1.2*((9.46568e-11)*np.power(T,4) - (1.20155e-7)*np.power(T,3) + (5.73768e-5)*np.square(T) - (1.22266e-2)*T + 9.82422e-1) # HS DLS Buffer Viscosity from SEDNTERP1, multiplied by 1.2 for D2O viscosity
    eta_HSBuffer_D2O = eta_HSBuffer*((5.94701e-8)*np.power(T,4) -(7.27430e-5)*np.power(T,3) +(3.3361e-2)*np.square(T) -(6.80129e0)*T + 5.21456e2)
    #eta_HSBuffer_D2O = 1.0544*eta_HSBuffer*((T - 273.15) - 6.498)
    
    #eta_NSBuffer = (-6.20268e-9)*np.power(T,3) + (5.97217e-6)*np.square(T) - (1.92772e-3)*T + 2.09153e-1 # No salt DLS Buffer Viscosity from SEDNTERP
    #Dt = (kB*T)/(6*np.pi*eta_HSBuffer*Rh)
    Dt = (kB*T)/(6*np.pi*eta_HSBuffer_D2O*Rh)
    #Dt = (kB*T)/(6*np.pi*eta_NSBuffer*Rh)
    
    return Dt
    

### Minimization function for global fit
def objective(fit_params, x, data_WT):
    
    if x == 0:

        resid = np.array([])
        for ind, group in data_WT:
            D_dict, c_dict, K_dict, P_dict = supercageNucPol(fit_params, group)
            resid = np.append(resid,np.array(group.D.values*1e4) - D_dict['Dz']*1e4)    
    
    if x == 1:
        
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.set_ylim(1.5e-11,8.5e-11)
        
        resid = np.array([])
        for ind, group in data_WT:
            D_dict, c_dict, K_dict, P_dict = supercageNucPol(fit_params, group)
            resid = np.append(resid,np.array(group.D.values*1e4) - D_dict['Dz']*1e4)
        
            ax.plot(group.Temperature,group.D.values,'ko')
            ax.plot(group.Temperature,D_dict['Dz'],'r')
    
        plt.ion()
        plt.pause(0.00001)
        plt.ioff()
        plt.close()
    
    #print("RSS: ",np.sum(np.square(resid)))
    #for k in fit_params.keys():
    #print(fit_params[k])
    #fit_params.pretty_print(fmt='e',colwidth=12,columns=['value'])
    
    return resid
  
    
### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(data_WT, opt_params):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DLSglobalfits.pdf')
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    cmap = plt.cm.PuRd(np.linspace(0,1,len(data_WT)*3))
    hist_colors = ['#2166ac','#92c5de','#d1e5f0','#d6604d','#b2182b',]
    
    D_fig = plt.figure()
    D_ax = D_fig.add_subplot(111)
    D_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    D_ax.tick_params(direction='in',axis='both',length=3,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    D_ax.yaxis.major.formatter._useMathText = True
    D_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
    D_ax.set_ylabel('$D_{obs.}$ $cm^{2}$ $s^{-1}$',fontsize=14)
    D_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    #D_ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    
    Cooperativity_fig = plt.figure()
    Cooperativity_ax = Cooperativity_fig.add_subplot(111)
    Cooperativity_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    Cooperativity_ax.tick_params(direction='in',axis='both',length=3,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    Cooperativity_ax.yaxis.major.formatter._useMathText = True
    Cooperativity_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
    Cooperativity_ax.set_ylabel('$log_{10}(\sigma)$',fontsize=14)
    Cooperativity_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    
    K_fig = plt.figure()
    K_ax = K_fig.add_subplot(111)
    K_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    K_ax.tick_params(direction='in',axis='both',length=3,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    K_ax.yaxis.major.formatter._useMathText = True
    K_ax.set_xlabel("1/T $K^{-1}$",fontsize=14)
    K_ax.set_ylabel('ln(K)',fontsize=14)
    K_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    
    hist_temperatures = np.array([10,20,30,40,50])
    
    cidx = 0
    for ind, group in data_WT:
        
        sim_df = pd.DataFrame()
        sim_df['Temperature'] = np.linspace(np.min(group.Temperature),np.max(group.Temperature),100)
        sim_df['Concentration'] = np.zeros(len(sim_df['Temperature']))
        sim_df['Concentration'] = group.Concentration.iloc[0]
        D_dict, c_dict, K_dict, P_dict = supercageNucPol(opt_params, sim_df)

        D_ax.plot(group.Temperature,group.D*1e4,'o',color=cmap[cidx])
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['Dz'])*1e4,color=cmap[cidx])
        D_ax.plot(sim_df['Temperature'],np.array(D_dict["D3_0"])*1e4,'--',color='#4292c6')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D16_0"])*1e4,'--',color='#78c679')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*opt_params['N'].value}_0"])*1e4,'k--')
        
        #c_fig = plt.figure()
        #c_ax = c_fig.add_subplot(111)
        #c_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        #c_ax.yaxis.major.formatter._useMathText = True
        #c_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
        #c_ax.set_ylabel('Concentration $\mu$M',fontsize=14)
        #c_ax.set_title(f"DegP_2, $[M]_{{T}}$: {group.Concentration.iloc[0]} $\mu$M",fontsize=14)
        #c_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        #c_ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        
        Cooperativity_ax.plot(sim_df['Temperature'],np.log10(K_dict['Kn1']/K_dict['Kn2']),LineWidth=2,color='#fc8d59',label='$K_{n,1}$/$K_{n,2}$')
        Cooperativity_ax.plot(sim_df['Temperature'],np.log10(K_dict['Kn2']/K_dict['Ke']),LineWidth=2,color='#df65b0',label='$K_{n,2}$/$K_{e}$')
        K_ax.plot(1./(sim_df['Temperature'] + 273.15),np.log(K_dict['Kn1']),LineWidth=2,color='k',label='$K_{n,1}$')
        K_ax.plot(1./(sim_df['Temperature'] + 273.15),np.log(K_dict['Kn2']),LineWidth=2,color='r',label='$K_{n,2}$')
        K_ax.plot(1./(sim_df['Temperature'] + 273.15),np.log(K_dict['Ke']),LineWidth=2,color='b',label='$K_{e}$')
        
        hist_figs = [plt.figure() for x in range(len(hist_temperatures))]
        Phist_figs = [plt.figure() for x in range(len(hist_temperatures))]
        Phist_axs = [Phist_figs[x].add_subplot(111) for x in range(len(hist_temperatures))]
        hist_axs = [hist_figs[x].add_subplot(111) for x in range(len(hist_temperatures))]
        for x in range(len(hist_temperatures)):
            hist_axs[x].tick_params(direction='in',axis='both',length=3,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
            hist_axs[x].yaxis.major.formatter._useMathText = True
            hist_axs[x].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
            hist_axs[x].set_xlabel("Oligomer size",fontsize=14)
            hist_axs[x].set_xlim([0,101])
            hist_axs[x].set_ylabel('Particle concentration $\mu$M',fontsize=14)
            
            Phist_axs[x].tick_params(direction='in',axis='both',length=3,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
            Phist_axs[x].yaxis.major.formatter._useMathText = True
            Phist_axs[x].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
            Phist_axs[x].set_xlabel("Oligomer size",fontsize=14)
            Phist_axs[x].set_xlim([0,101])
            Phist_axs[x].set_ylabel('Population',fontsize=14)
            
        # Plot concentration histograms
        # Temperature indices
        for hcindex, temperature in enumerate(hist_temperatures):

            tindex = np.argmin(np.abs(temperature - sim_df['Temperature'].values))
                
            for x in range(1,int(opt_params['N'].value)+1):
            
                if x == 2:
                    
                    hist_axs[hcindex].bar(3*x,c_dict[f"c1{3*x}"][tindex]*1e6,color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                    hist_axs[hcindex].bar(3.5*x,c_dict[f"c2{3*x}"][tindex]*1e6,color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                    Phist_axs[hcindex].bar(3*x,P_dict[f"P1{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                    Phist_axs[hcindex].bar(3.5*x,P_dict[f"P2{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                    
                else:
                        
                    hist_axs[hcindex].bar(3*x,c_dict[f"c{3*x}"][tindex]*1e6,color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                    Phist_axs[hcindex].bar(3*x,P_dict[f"P{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
            
            hist_axs[hcindex].set_title('{}, $M_{{T}}$ = {} $\mu$M, {} \N{DEGREE SIGN}C'.format(group.Sample.iloc[0],group.Concentration.iloc[0],temperature))
            Phist_axs[hcindex].set_title('{}, $M_{{T}}$ = {} $\mu$M, {} \N{DEGREE SIGN}C'.format(group.Sample.iloc[0],group.Concentration.iloc[0],temperature))
                     
            pdf.savefig(hist_figs[hcindex])
            pdf.savefig(Phist_figs[hcindex])
            plt.close(hist_figs[hcindex])
            plt.close(Phist_figs[hcindex])
    
        cidx += 2
        
        
#        for x in range(1,int(opt_params['N'].value)+1):
#            if x == 1:
#                
#                D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}_0"])*1e4,label=f"$D_{3*x,0}$")
#                #D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}"])*1e4,'--',label=f"$D_{3*x}$")
#                #print(f"D{3*x}_0: ",D_dict[f"D{3*x}_0"])
#                #c_ax.plot(sim_df['Temperature'],np.array(c_dict[f"c{3*x}"])*1e6,LineWidth=2,label=f"$c_{3*x}$")
#            
#            if x == 2:
#                
#                # Two hexamers need to be plotted
#                D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D1{3*x}_0"])*1e4,label=f"$D_1{3*x,0}$")
#                #D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D2{3*x}_0"])*1e4,label=f"$D_2{3*x,0}$")
#                
#                #c_ax.plot(sim_df['Temperature'],np.array(c_dict[f"c1{3*x}"])*1e6,LineWidth=2,label=f"$c_1{3*x}$")
#                #c_ax.plot(sim_df['Temperature'],np.array(c_dict[f"c2{3*x}"])*1e6,LineWidth=2,label=f"$c_2{3*x}$")
#                
#            if x > 2:
#                
#                #c_ax.plot(sim_df['Temperature'],np.array(c_dict[f"c{3*x}"])*1e6,LineWidth=2,label=f"$c_{3*x}$")
#                pass
#            
#            if x == opt_params['N'].value:
#                
#                D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}_0"])*1e4,label=f"$D_{3*x,0}$")
#                #D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}"])*1e4,'--',label=f"$D_{3*x}$")
#                #c_ax.plot(sim_df['Temperature'],np.array(c_dict[f"c{3*x}"])*1e6,LineWidth=2,label=f"$c_{3*x}$")
#                #print(f"### The concentration of the {3*opt_params['N'].value}-mer at PT = {group.Concentration.iloc[0]} uM as a function of temperature is: ####")
#                #print(c_dict[f"c{3*x}"])
#                #print('\n')
            
        #c_ax.legend(loc='upper left',fontsize=12,frameon=False)
        #K_ax.legend(loc='upper left',fontsize=12,frameon=False)
        #Cooperativity_ax.legend(loc='upper left',fontsize=12,frameon=False)
        #D_ax.legend(loc='upper left',fontsize=12,frameon=False)
        #pdf.savefig(c_fig)
        #pdf.savefig(hist_fig)
        #plt.close(c_fig)
        #plt.close(hist_fig)
    
    pdf.savefig(K_fig)
    plt.close(K_fig)
    pdf.savefig(Cooperativity_fig)
    plt.close(Cooperativity_fig)
    pdf.savefig(D_fig)
    plt.close(D_fig)
    pdf.close()

R = 1.9872e-3 # Gas constant for calculating affinities
R = R*4.184*1000 # Thermodynamic parameters have to be in J/mol and J/mol*K to match this gas constant
main()