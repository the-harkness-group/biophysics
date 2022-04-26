#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:00:38 2020

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
import timeit

### Read in data and set up for fitting
def main():
    
    data = pd.read_csv(sys.argv[1])

    ### Set up parameters and groups    
    data_WT = data[data['Sample'] == 'P2'] # DegP protease-dead, WT
    #data_WT = data_WT[data_WT['Temperature'] <= 30]
    groups_WT = data_WT.groupby('Concentration')
    
    fit_params = Parameters()
    
    fit_params.add('dHn_0', value=-188000,vary=False)
    fit_params.add('Kn_0', value=50000,vary=False)
    fit_params.add('dCpn', value=-8000,vary=False)
    
    fit_params.add('dHe_0', value=30000)
    fit_params.add('Ke_0', value=100000)
    fit_params.add('dCpe', value=-3000,vary=False)

    fit_params.add('T0', value=30+273.15, vary=False)
    
    fit_params.add('Rh_3', value=4.9e-9)
    fit_params.add('m',value=-1/3)
    fit_params.add('n',value=2)
    fit_params.add('N',value=100)
    fit_params.add('k_d',value=5)
        
    # Simulate data
    plot_fit(groups_WT, fit_params)

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
    
    k_d = fit_params['k_d'].value
    
    n = fit_params['n'].value
    N = int(fit_params['N'].value)
    m = fit_params['m'].value
    
    # Set up dictionaries for species up to size N
    D_dict = {}
    c_dict = {}
    for x in range(1,N+1):
        D_dict[f"D{3*x}_0"] = np.array([])
        c_dict[f"c{3*x}"] = np.array([])
    
    D_dict['Dz'] = np.array([])
    K_dict = {'Kn':np.array([]),'Ke':np.array([])}
    
    #start_time = timeit.default_timer()
    for y in range(len(Temperatures)):
            
        # Calculate equilibrium constants
        Kn = Kn_0*np.exp((dHn_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpn/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        Ke = Ke_0*np.exp((dHe_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpe/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))

######### Solve explicit populations       
#        solutions = np.array([])
#        p = [0.01 for x in range(1,N+1)]
#        q = [Concentrations[0]]
#        for x in range(1,N+1):
#            if x <= n-1:
#                q.append(Kn)
#            elif x > n-1:
#                q.append(Ke)
#        ffs_partial = partial(ffs4,q,fit_params)
#        
#        #start_time = timeit.default_timer()
#        solutions = np.append(solutions,opt.root(ffs_partial,p,method='lm'))
#        #elapsed = timeit.default_timer() - start_time
#        #print('Solve time: ',elapsed)
#        
#        D_num = 0
#        D_numtemp = 0
#        D_den = 0
#        D_dentemp = 0
#        D3_0 = StokesEinstein(Temperatures[y], fit_params['Rh_3'].value)
#        
#        #start_time = timeit.default_timer()
#        for idx,sol in zip(range(1,N+1),solutions):
#            c3x = sol.x[idx-1]*Concentrations[0]/(3*idx)
#            c_dict[f"c{3*idx}"] = np.append(c_dict[f"c{3*idx}"],c3x)
#            
#            D3x_0 = D3_0*idx**(m)
#            D_numtemp = (idx**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
#            D_num = D_num + D_numtemp
#            D_dentemp = (idx**2)*c3x
#            D_den = D_den + D_dentemp
#            D_dict[f"D{3*idx}_0"] = np.append(D_dict[f"D{3*idx}_0"],D3x_0)
#            
#        #elapsed = timeit.default_timer() - start_time      
        #print('Loop time: ',elapsed)
####################
        
        ################## Solve dimensionless trimer concentration
        constants = [(Concentrations[0]*Ke/3),Kn/Ke,n] # XT, sigma, nucleus size
        #a = 6*Ke
        #b = 3
        #c = -Concentrations[0]
        #c3_guess = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        #g_guess = c3_guess*Ke
        
        #roots = np.roots([9*Kn*Ke, 6*Kn, 3, -Concentrations[0]])
        #real_roots = np.real(roots[np.isreal(roots)])
        #g_guess = real_roots[real_roots > 0]
        g_guess = 1e-5
        ffs_partial = partial(ffs3,constants)
        sol = opt.root(ffs_partial,g_guess,method='lm')
        c3 = sol.x[0]/Ke
        #g = opt.fsolve(ffs3,g_guess,args=constants) # Guess initial dimensionless trimer concentration from equilibrium
        #c3 = g/Ke
        
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
                #c3x = np.power(Kn,(x-1))*np.power(c3,x)
                c3x = Kn*c_dict[f"c{3*(x-1)}"][-1]*c3
                c_dict[f"c{3*x}"] = np.append(c_dict[f"c{3*x}"],c3x)
                #print('x:',x,f"c{3*x}: ",c3x,'\n')
                #print('Knpower: ',Kn**(x-1),'c3xpower: ',c3**x,'\n')
                
            elif x > n:
                #c3x = np.power(Kn,(n-1))*np.power(Ke,(x-n))*np.power(c3,x)
                c3x = Ke*c_dict[f"c{3*(x-1)}"][-1]*c3
                c_dict[f"c{3*x}"] = np.append(c_dict[f"c{3*x}"],c3x)
                #print('x:',x,f"c{3*x}: ",c3x,'\n')
                #print('Kpower: ',(Kn**(n-1))*(Ke**(x-n)),'c3xpower: ',(c3**x))
            
            # Calculate the numerator and denominator terms and finally the average diffusion coefficient
            D3x_0 = D3_0*x**(m)
            D_numtemp = (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
            D_num = D_num + D_numtemp
            D_dentemp = (x**2)*c3x
            D_den = D_den + D_dentemp
            D_dict[f"D{3*x}_0"] = np.append(D_dict[f"D{3*x}_0"],D3x_0)
        
        # Make dictionary of D values for fit
        Dz = D_num/D_den
        D_dict['Dz'] = np.append(D_dict['Dz'],Dz)

        # Make dictionary of the K values
        K_dict['Kn'] = np.append(K_dict['Kn'],Kn)
        K_dict['Ke'] = np.append(K_dict['Ke'],Ke)
      
    #elapsed = timeit.default_timer() - start_time
    #print('Group time: ',elapsed)
    return D_dict, c_dict, K_dict

### Nucleated polymerization solver - dimensionless
#def ffs3(g,constants):
def ffs3(constants,g):
    
    XT, s, n = constants # Unpack constants
    
    first_term = (s**(-1))*((((s*g)**(n+1))*(n*s*g - n - 1) + s*g)/((s*g - 1)**2))
    second_term = -(s**(n - 1))*(((g**(n+1))*(n*g - n - 1))/((g - 1)**2))
    eq1 = -XT + (first_term + second_term)
    
    return eq1

### Nucleated polymerization solver - explicit populations
def ffs4(q,fit_params,p):
    
    N = int(fit_params['N'].value)
    n = fit_params['n'].value
        
    equations  = [-1+np.sum(p)]

    for x in range(2,N+1):
        if x <= n:
            equilibrium = (p[x-1]/(p[x-2]*p[0]*q[0]))*((3*(x - 1))/x)
            equations.append(q[x-1] - equilibrium)
        if x > n:
            equilibrium = (p[x-1]/(p[x-2]*p[0]*q[0]))*((3*(x - 1))/x)
            equations.append(q[x-1] - equilibrium)
    
    return equations

### Calculate diffusion coefficients
def StokesEinstein(T, Rh):
    
    kB = 1.38065e-23
    #eta = (2.414e-5)*10**(247.8/(T-140)) # Water viscosity
    #Dt = (kB*T)/(6*np.pi*eta*Rh)
    eta_HSBuffer = (9.46568e-11)*np.power(T,4) - (1.20155e-7)*np.power(T,3) + (5.73768e-5)*np.square(T) - (1.22266e-2)*T + 9.82422e-1 # HS DLS Buffer Viscosity from SEDNTERP1
    Dt = (kB*T)/(6*np.pi*eta_HSBuffer*Rh)
    
    return Dt

### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(data_WT, opt_params):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DLSglobalfits.pdf')
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = int(opt_params['N'].value)
    
    D_fig = plt.figure()
    D_ax = D_fig.add_subplot(111)
    D_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    D_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    D_ax.yaxis.major.formatter._useMathText = True
    D_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
    D_ax.set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$',fontsize=14)
    D_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    D_ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for ind, group in data_WT:
        
        sim_df = pd.DataFrame()
        sim_df['Temperature'] = np.linspace(np.min(group.Temperature),np.max(group.Temperature),100)
        sim_df['Concentration'] = np.zeros(len(sim_df['Temperature']))
        sim_df['Concentration'] = group.Concentration.iloc[0]
        D_dict, c_dict, K_dict = supercageNucPol(opt_params, sim_df)
        
        print(c_dict['c3'],c_dict['c9'])

        D_ax.plot(group.Temperature,group.D*1e4,'ko',label='Experiment')
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['Dz'])*1e4,'r',label='Fit')
        
        c_fig = plt.figure()
        c_ax = c_fig.add_subplot(111)
        #c_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        c_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        c_ax.yaxis.major.formatter._useMathText = True
        c_ax.set_xlabel("Temperature \N{DEGREE SIGN}C",fontsize=14)
        c_ax.set_ylabel('Concentration $\mu$M',fontsize=14)
        c_ax.set_title(f"DegP_2, $[M]_{{T}}$: {group.Concentration.iloc[0]} $\mu$M",fontsize=14)
        c_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        c_ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        
        for x in range(1,int(opt_params['N'].value)+1):
            #if 1 <= x <= 10:
                #D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}_0"])*1e4,label=f"$D_{3*x}$")
                #print(f"c{3*x}",c_dict[f"c{3*x}"])
            #elif x == opt_params['N'].value:
                #D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*x}_0"])*1e4,label=f"$D_{3*x}$")
    
            c_ax.plot(sim_df['Temperature'],np.array(c_dict[f"c{3*x}"])*1e6,LineWidth=2,label=f"$c_{3*x}$")
        
        pdf.savefig(c_fig)
        plt.close(c_fig)
        
    pdf.savefig(D_fig)
    pdf.close()
    
R = 1.9872e-3 # Gas constant for calculating affinities
R = R*4.184*1000 # Thermodynamic parameters have to be in J/mol and J/mol*K to match this gas constant
main()