#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 15:00:51 2020

@author: robertharkness
"""

import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
from lmfit import minimize, Parameters, report_fit
from scipy import optimize as opt
from functools import partial
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
import customcolormap
import matplotlib.colors as mcolors
import time

### Read in data and set up for fitting
def main():
    
    params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(params['Dataset']) # Get dataset name to fit
    fit_data = data[data.Sample == params['Sample']] # Extract sample data to fit from dataset in cases where data might be for multiple samples
    fit_data = fit_data[fit_data.Temperature <= 50.5] # Select temperature range, DLS plate reader sometimes goes slightly above 50C e.g. 50.1 so extend a little beyond 50 here
    fit_data = fit_data[fit_data.Concentration.isin(params['Fit params']['Concentrations'])] # Select fit concentrations
    fit_groups = fit_data.groupby('Concentration') # Make fitting groups by total protein concentration
    
    fit_params = Parameters() # Set up fit/simulation parameters

    fit_params.add('dHn1_0', value=params['Fit params']['dHn1_0']['Value'],vary=params['Fit params']['dHn1_0']['Vary']) # dH1 at ref temp.
    fit_params.add('Kn1_0', value=params['Fit params']['Kn1_0']['Value'],vary=params['Fit params']['Kn1_0']['Vary']) # K1 at ref temp.
    fit_params.add('dCpn1', value=params['Fit params']['dCpn1']['Value'],vary=params['Fit params']['dCpn1']['Vary']) # K1 dCp
    
    fit_params.add('dHn2_0', value=params['Fit params']['dHn2_0']['Value'],vary=params['Fit params']['dHn2_0']['Vary']) # dH2 at ref temp.
    fit_params.add('Kn2_0', value=params['Fit params']['Kn2_0']['Value'],vary=params['Fit params']['Kn2_0']['Vary']) # K2 at ref temp.
    fit_params.add('dCpn2', value=params['Fit params']['dCpn2']['Value'],vary=params['Fit params']['dCpn2']['Vary']) # K2 dCp
    
    #fit_params.add('dHe_0', value=params['Fit params']['dHe_0']['Value'],vary=params['Fit params']['dHe_0']['Vary']) # dH3 at ref temp.
    #fit_params.add('Ke_0', value=params['Fit params']['Ke_0']['Value'],vary=params['Fit params']['Ke_0']['Vary']) # K3 at ref temp.
    #fit_params.add('dCpe', value=params['Fit params']['dCpe']['Value'],vary=params['Fit params']['dCpe']['Vary']) # K3 dCp

    fit_params.add('T0', value=params['Fit params']['T0']['Value']+273.15,vary=params['Fit params']['T0']['Vary']) # Reference temperature
    
    fit_params.add('Rh_3', value=params['Fit params']['Rh_3']['Value'], vary=params['Fit params']['Rh_3']['Vary'],min=4e-9,max=6e-9) # Trimer hydrodynamic radius
    fit_params.add('h_scale',value=params['Fit params']['h_scale']['Value'],vary=params['Fit params']['h_scale']['Vary'],min=-0.6,max=-0.1) # >6-mer Diffusion Constant Scaling
    fit_params.add('j_scale',value=params['Fit params']['j_scale']['Value'],vary=params['Fit params']['j_scale']['Vary']) # 3-6-mer Diffusion Constant Scaling
    fit_params.add('n',value=params['Fit params']['n']['Value'],vary=params['Fit params']['n']['Vary'],min=2,max=50) # Nucleus size
    fit_params.add('N',value=params['Fit params']['N']['Value'],vary=params['Fit params']['N']['Vary']) # Final oligomer size
    fit_params.add('k_d0',value=params['Fit params']['k_d0']['Value'],vary=params['Fit params']['k_d0']['Vary']) # Effective nonideality constant
    fit_params.add('m',value=params['Fit params']['m']['Value'],vary=params['Fit params']['m']['Vary']) # Effective nonideality constant temperature dependence
    
    fit_params.add('eta1',value=params['Eta params'][0],vary=False) # Viscosity parameters from fitting third order polynomial to SEDNTERP viscosity as a function of temperature
    fit_params.add('eta2',value=params['Eta params'][1],vary=False) # a*T^3 + b*T^2 + c*T + d, a = eta params[0], b = eta params[1], ...
    fit_params.add('eta3',value=params['Eta params'][2],vary=False)
    fit_params.add('eta4',value=params['Eta params'][3],vary=False)

    run_fit = params['Run fit'] # Run fit with 'y', simulate data with 'n'
    plot_minimization = params['Plot minimization'] # If running fit, plot minimization in real time with 'y', don't plot with 'n'
    PDFname = params['Output PDF'] # Name for fit or simulation output PDF with plots
    samplename = params['Sample name'] # Name of sample for plot text
    
    if run_fit == 'y':
        
        # Fit data    
        result = minimize(objective, fit_params, method='nelder', args=(plot_minimization, fit_groups))
        report_fit(result)
        opt_params = result.params
        plot_fit(fit_groups, opt_params, PDFname, samplename)
        
        print('\n')
        opt_params.pretty_print(fmt='e',colwidth=12,columns=['value']) # Print nice parameter table
        
        print('\nYOU HAVE KE SET EQUAL TO K2!!!!!!!!!!!!!!!!!!!!!!!!')
        
    if run_fit == 'n':
        
        # Simulate data
        plot_fit(fit_groups, fit_params, PDFname, samplename)
        fit_params.pretty_print(fmt='e',colwidth=12,columns=['value'])
        
        print('\nYOU HAVE KE SET EQUAL TO K2!!!!!!!!!!!!!!!!!!!!!!!!')


### Nucleated polymerization, two pathways
def supercageNucPol(fit_params, fit_group):
    
    Temperatures = np.array(fit_group.Temperature + 273.15)
    Concentrations = np.array(fit_group.Concentration*1e-6)
    
    # Set up thermodynamic and hydrodynamic parameters
    dHn1_0 = fit_params['dHn1_0'].value
    Kn1_0 = abs(fit_params['Kn1_0'].value)
    dCpn1 = fit_params['dCpn1'].value
    
    dHn2_0 = fit_params['dHn2_0'].value
    Kn2_0 = abs(fit_params['Kn2_0'].value)
    dCpn2 = fit_params['dCpn2'].value
    
    #dHe_0 = fit_params['dHe_0'].value
    #Ke_0 = abs(fit_params['Ke_0'].value)
    #dCpe = fit_params['dCpe'].value
    
    dHe_0 = fit_params['dHn2_0'].value
    Ke_0 = abs(fit_params['Kn2_0'].value)
    dCpe = fit_params['dCpn2'].value
    
    m = fit_params['m'].value # Nonideality
    k_d0 = fit_params['k_d0'].value
    
    n = int(round(fit_params['n'].value))
    N = int(fit_params['N'].value)
    h_scale = fit_params['h_scale'].value
    j_scale = fit_params['j_scale'].value
    
    eta1 = fit_params['eta1'].value # Viscosity polynomial parameters from SEDNTERP
    eta2 = fit_params['eta2'].value
    eta3 = fit_params['eta3'].value
    eta4 = fit_params['eta4'].value
    
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
    
    X_guess = 1e-5 # Initial guess for dimensionless trimer concentration
    t0 = time.time()
    for y in range(len(Temperatures)):
        
        # Calculate equilibrium constants
        Kn1 = Kn1_0*np.exp((dHn1_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpn1/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        Kn2 = Kn2_0*np.exp((dHn2_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpn2/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        Ke = Ke_0*np.exp((dHe_0/R)*((1/fit_params['T0'].value) - (1/Temperatures[y])) + (dCpe/R)*(np.log(Temperatures[y]/fit_params['T0'].value) + (fit_params['T0'].value/Temperatures[y]) - 1))
        
        ################## Solve dimensionless trimer concentration ##################
        constants = [(Concentrations[0]*Ke/3),Kn1/Kn2,Kn2/Ke,n] # XT, alpha1 = Kn1/Kn2, alpha2 = Kn2/Ke, nucleus size
        ffs_partial = partial(ffs3,constants)        
        sol = opt.root(ffs_partial,X_guess,method='lm')        
        c3 = sol.x[0]/Ke # Trimer concentration in molar
        
        # Calculate average oligomer size for N+1 to infinity
        #x3 = c3*Ke # dimensionless trimer concentration
        #ff = np.power((Kn2/Ke),(n-1))*(np.power(x3,N+1)/(1 - x3)) # dimensionless fiber concentration N+1 to infinity
        #zf = np.power(Kn2/Ke,n-1)*(-(np.power(x3,N+1)*(N*x3-N-1))/np.square(x3-1)) # dimensionless Z concentration N+1 to infinity
        #fl = zf/ff # average fiber length for N+1 to infinity is Z/F
        #print(f"The average fiber length above size N is : {fl}")
        
        # Calculate temperature-dependent non-ideality constant
        k_d = k_d0 + m*(Temperatures[y] - fit_params['T0'].value)

        # Generate populations and concentrations from solver solutions, calculate diffusion coefficients
        D_num = 0
        D_numtemp = 0
        D_den = 0
        D_dentemp = 0
        D3_0 = StokesEinstein(Temperatures[y], eta1, eta2, eta3, eta4, fit_params['Rh_3'].value) # Calculate trimer diffusion constant first
        
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
        
    t1 = time.time()
    print(f"The total time is: {t1-t0}")
        
    return D_dict, c_dict, K_dict, P_dict


### Nucleated polymerization solver, two pathways - dimensionless
def ffs3(constants,X):
    
    XT, a1, a2, n = constants # Unpack constants
    
    first_term = (a2**(-1))*((((a2*X)**(n+1))*(n*a2*X - n - 1) + a2*X)/((a2*X - 1)**2))
    second_term = -(a2**(n - 1))*(((X**(n+1))*(n*X - n - 1))/((X - 1)**2))
    eq1 = -XT + 2.*a1*a2*np.square(X) + first_term + second_term
    
    return eq1


### Calculate diffusion coefficients
def StokesEinstein(Temperature, eta1, eta2, eta3, eta4, Rh):
    
    eta = eta1*np.power(Temperature,3) + eta2*np.square(Temperature) + eta3*Temperature + eta4 # Third order polynomial viscosity as a function of temperature from SEDNTERP
    Dt = (kB*Temperature)/(6*np.pi*eta*Rh)

    return Dt
    

### Minimization function for global fit
def objective(fit_params, plot_minimization, fit_groups):
    
    if plot_minimization == 'n': # Don't plot fit in real time

        resid = np.array([])
        for ind, group in fit_groups:
            
            D_dict, c_dict, K_dict, P_dict = supercageNucPol(fit_params, group)            
            resid = np.append(resid,np.array(group.D.values*1e4) - D_dict['Dz']*1e4)
    
    if plot_minimization == 'y': # Plot fit in real time
        
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.set_ylim(1.5e-11,8.5e-11)
        
        resid = np.array([])
        for ind, group in fit_groups:
            D_dict, c_dict, K_dict, P_dict = supercageNucPol(fit_params, group)
            resid = np.append(resid,np.array(group.D.values*1e4) - D_dict['Dz']*1e4)
        
            ax.plot(group.Temperature,group.D.values,'ko')
            ax.plot(group.Temperature,D_dict['Dz'],'r')
    
        plt.ion()
        plt.pause(0.00001)
        plt.ioff()
        plt.close()
    
    return resid
  
    
### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(fit_groups, opt_params, PDFname, samplename):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(PDFname)
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.style.use('figure')
    #cmap = plt.cm.PuRd(np.linspace(0,1,len(fit_groups)*7))
    hist_colors = ['#2166ac','#92c5de','#fddbc7','#d6604d','#b2182b',]
    hex_colors = ['#020101','#281112','#4E2128','#733243','#984364','#BD548A','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF']
    hex_colors = list(reversed(hex_colors))
    #cmap = ['#020101','#281112','#4E2128','#733243','#984364','#BD548A','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF']
    #cmap = list(reversed(cmap))
    
    rgb_cmap = customcolormap.get_continuous_cmap(hex_colors,fit_groups.ngroups) #  Get rgb map at specified number of points using the above HEX colors
    cmap = [mcolors.to_hex(rgb_cmap(a)) for a in range(fit_groups.ngroups)] # Convert rgb map back into HEX colors for indexing at number of points in dataset
                   
    D_fig = plt.figure()
    D_ax = D_fig.add_subplot(111)
    D_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    D_ax.yaxis.major.formatter._useMathText = True
    D_ax.set_xlabel("Temperature \N{DEGREE SIGN}C")
    D_ax.set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$')
    D_axins = inset_axes(D_ax,width="37%",height="37%",loc=2,borderpad=1.0)
    
    D_fig2, D_ax2 = plt.subplots(1,figsize=(3,5))
    D_ax2.yaxis.major.formatter._useMathText = True
    D_ax2.set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$')
    D_ax2.bar([1],[2.33e-7],width=0.5,color='#fa9fb5',label='experiment') # Experiment vs HYDROPRO diffusion constants for hexamer
    D_ax2.bar([2],[2.35e-7],width=0.5,color='#fa9fb5',label='HYDROPRO',hatch='///',edgecolor='w') 
    D_ax2.bar([3],[2.70e-7],width=0.5,color='#9ebcda',label='experiment') # Experiment vs HYDROPRO diffusion constants for trimer
    D_ax2.bar([4],[2.75e-7],width=0.5,color='#9ebcda',label='HYDROPRO',hatch='///',edgecolor='w')
    D_ax2.set_xticks([1,2,3,4])
    D_ax2.set_xticklabels(['$M_{6,expt.}$','$M_{6,calc.}$','$M_{3,expt.}$','$M_{3,calc.}$'],rotation=60)
              
    K_fig = plt.figure()
    K_ax = K_fig.add_subplot(111)
    K_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
    K_ax.yaxis.major.formatter._useMathText = True
    K_ax.xaxis.major.formatter._useMathText = True
    K_ax.set_xlabel("1/T $K^{-1}$")
    K_ax.set_ylabel('ln(K)')
   
    hist_temperatures = np.array([10,20,30,40,50]) # Temperatures for plotting particle histograms
    #pop_plot_concs = [11, 192] # Total monomer concentrations for plotting histograms # LA REFOLDED DegP2 25 mM NaH2PO4, 200 mM NaCl, 1 mM EDTA, pH 7.0
    #pop_plot_concs = [12.9, 193.1] # Total monomer concentrations for plotting histograms # Rob REFOLDED DegP2 25 mM NaH2PO4, 1 mM EDTA, pH 7.0
    pop_plot_concs = [10, 203] # Total monomer concentrations for plotting histograms # LA REFOLDED DegP3 25 mM NaH2PO4, 200 mM NaCl, 1 mM EDTA, pH 7.0
    #pop_plot_concs = [11, 196] # Total monomer concentrations for plotting histograms # LA REFOLDED DegP9 25 mM NaH2PO4, 200 mM NaCl, 1 mM EDTA, pH 7.0
    
    cidx = 0
    for ind, group in fit_groups:
        
        sim_df = pd.DataFrame()
        sim_df['Temperature'] = np.linspace(np.min(group.Temperature),np.max(group.Temperature),100)
        sim_df['Concentration'] = np.zeros(len(sim_df['Temperature']))
        sim_df['Concentration'] = group.Concentration.iloc[0]
        D_dict, c_dict, K_dict, P_dict = supercageNucPol(opt_params, sim_df)

        D_ax.plot(group.Temperature,group.D*1e4,'o',color=cmap[cidx])
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['Dz'])*1e4,color=cmap[cidx])
        D_ax.plot(sim_df['Temperature'],np.array(D_dict["D3_0"])*1e4,'--',color='#9ebcda') # 3-mer
        D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D16_0"])*1e4,'--',color='#fa9fb5') # 6-mer
        D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*8}_0"])*1e4,'--',color='#49006a') # 24-mer
        #D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*opt_params['n'].value}_0"])*1e4,'--',color='#8c6bb1') # nucleus size-mer
        #D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*opt_params['N'].value}_0"])*1e4,'k--') # N-mer final size
            
        # Plot concentration histograms
        # Temperature indices
        if ind in pop_plot_concs:
            
            #hist_figs = [plt.figure() for x in range(len(hist_temperatures))]
            Phist_figs = [plt.figure() for x in range(len(hist_temperatures))]
            Phist_axs = [Phist_figs[x].add_subplot(111) for x in range(len(hist_temperatures))]
            #hist_axs = [hist_figs[x].add_subplot(111) for x in range(len(hist_temperatures))]
            
            for x in range(len(hist_temperatures)):
                #hist_axs[x].tick_params(direction='in',axis='both',length=3,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
#               hist_axs[x].yaxis.major.formatter._useMathText = True
#               hist_axs[x].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
#               hist_axs[x].set_xlabel("Oligomer size",fontsize=14)
#               hist_axs[x].set_xlim([0,101])
#               hist_axs[x].set_ylabel('Particle concentration $\mu$M',fontsize=14)
            
                #Phist_axs[x].tick_params(direction='in',axis='both',length=3,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
                Phist_axs[x].yaxis.major.formatter._useMathText = True
                Phist_axs[x].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #Phist_axs[x].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
                Phist_axs[x].set_xlabel("Oligomer size")
                Phist_axs[x].set_xlim([0,41])
                Phist_axs[x].set_ylabel('Population')
            
            for hcindex, temperature in enumerate(hist_temperatures):

                tindex = np.argmin(np.abs(temperature - sim_df['Temperature'].values))
                
                for x in range(1,int(opt_params['N'].value)+1):
            
                    if x == 2: # For hexamer A and B
                    
                        #hist_axs[hcindex].bar(3*x,c_dict[f"c1{3*x}"][tindex]*1e6,color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                        #hist_axs[hcindex].bar(3.5*x,c_dict[f"c2{3*x}"][tindex]*1e6,color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                        Phist_axs[hcindex].bar(3*x,P_dict[f"P1{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                        Phist_axs[hcindex].bar(3.5*x,P_dict[f"P2{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                    
                    else:  # All other sizes
                        
                        #hist_axs[hcindex].bar(3*x,c_dict[f"c{3*x}"][tindex]*1e6,color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                        Phist_axs[hcindex].bar(3*x,P_dict[f"P{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
            
                #hist_axs[hcindex].set_title('{}, $M_{{T}}$ = {} $\mu$M, {} \N{DEGREE SIGN}C'.format(group.Sample.iloc[0],group.Concentration.iloc[0],temperature))
                Phist_axs[hcindex].text(0.95,0.95,'{}\n$M_{{T}}$ = {} $\mu$M\n{} \N{DEGREE SIGN}C'.format(samplename,group.Concentration.iloc[0],temperature),fontsize=20,va='top',ha='right',ma='left',transform=Phist_axs[hcindex].transAxes)
                #x0,x1 = Phist_axs[hcindex].get_xlim()
                #y0,y1 = Phist_axs[hcindex].get_ylim()
                #Phist_axs[hcindex].set_aspect(abs(x1-x0)/abs(y1-y0))
                
                #pdf.savefig(hist_figs[hcindex])
                Phist_figs[hcindex].tight_layout(pad=1)
                pdf.savefig(Phist_figs[hcindex])
                Phist_figs[hcindex].savefig(f"./Figures/Populations_{temperature}C_{group.Concentration.iloc[0]}uM.png")
                #plt.close(hist_figs[hcindex])
                plt.close(Phist_figs[hcindex])
            
        if ind == min(pop_plot_concs): # Lowest concentration for inset
            
            # Just plot equilibrium constants once since they are the same in every temperature range
            K_ax.plot(1./(sim_df['Temperature'] + 273.15),np.log(K_dict['Kn1']),linewidth=2,color='#4393c3',label='$K_{1}$')
            K_ax.plot(1./(sim_df['Temperature'] + 273.15),np.log(K_dict['Kn2']),linewidth=2,color='#f4a582',label='$K_{2}$')
            K_ax.plot(1./(sim_df['Temperature'] + 273.15),np.log(K_dict['Ke']),linewidth=2,color='#b2182b',label='$K_{3}$')
            K_ax.legend(frameon=False,loc='upper left')
            
            # Multiplying by 1e11 to make scale on inset match the full plot
            D_axins.plot(group.Temperature,group.D*1e11,'o',markersize=6,color=cmap[cidx])
            D_axins.plot(sim_df['Temperature'],np.array(D_dict['Dz'])*1e11,linewidth=2,color=cmap[cidx])
            D_axins.plot(sim_df['Temperature'],np.array(D_dict["D3_0"])*1e11,'--',linewidth=2,color='#9ebcda') # 3-mer
            D_axins.plot(sim_df['Temperature'],np.array(D_dict[f"D16_0"])*1e11,'--',linewidth=2,color='#fa9fb5') # 6-mer
            D_axins.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*8}_0"])*1e11,'--',linewidth=2,color='#49006a') # 24-mer
            D_axins.set_xticks([30, 40, 50])
            D_axins.set_yticks([5,6,7,8])
            D_axins.set_xlim([29.5,50.5])
            D_axins.set_ylim([4.8,8.5])
            D_axins.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
            D_axins.yaxis.major.formatter._useMathText = True
            
        cidx += 1
    
    K_fig.tight_layout(pad=1)
    pdf.savefig(K_fig)
    K_fig.savefig('./Figures/VantHoffplot.png',format='png')
    plt.close(K_fig)
    D_fig.tight_layout(pad=1)
    D_fig2.tight_layout(pad=1)
    pdf.savefig(D_fig)
    pdf.savefig(D_fig2)
    D_fig.savefig('./Figures/full_Dz.png',format='png')
    D_fig2.savefig('./Figures/Dbar.png',format='png')
    plt.close(D_fig)
    plt.close(D_fig2)
    pdf.close()

R = 1.9872e-3 # Gas constant for calculating affinities
R = R*4.184*1000 # Thermodynamic parameters have to be in J/mol and J/mol*K to match this gas constant
kB = 1.38065e-23 # Boltzmann constant for viscosity
main()