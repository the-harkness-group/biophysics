#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:51:30 2021

@author: robertharkness
"""

import sys
import os
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
import time
import pickle

### Read in data and set up for fitting
def main():
    
    params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(params['Dataset']) # Get dataset name to fit
    fit_data = data[data.Sample == params['Sample']] # Extract sample data to fit from dataset in cases where data might be for multiple samples
    fit_data = fit_data[fit_data.Temperature <= 50.5] # Select temperature range, DLS plate reader sometimes goes slightly above 50C e.g. 50.1 so extend a little beyond 50 here
    fit_data = fit_data[fit_data.Concentration.isin(params['Fit params']['Concentrations'])] # Select fit concentrations
    fit_groups = fit_data.groupby('Concentration') # Make fitting groups by total protein concentration
    
    fit_params = Parameters() # Set up fit/simulation parameters

    fit_params.add('dH1_0', value=params['Fit params']['dHn1_0']['Value'],vary=params['Fit params']['dHn1_0']['Vary']) # dH1 at ref temp.
    fit_params.add('K1_0', value=params['Fit params']['Kn1_0']['Value'],vary=params['Fit params']['Kn1_0']['Vary'],min=0) # K1 at ref temp.
    fit_params.add('dCp1', value=params['Fit params']['dCpn1']['Value'],vary=params['Fit params']['dCpn1']['Vary']) # K1 dCp
    
    fit_params.add('dH2_0', value=params['Fit params']['dHn2_0']['Value'],vary=params['Fit params']['dHn2_0']['Vary']) # dH2 at ref temp.
    fit_params.add('K2_0', value=params['Fit params']['Kn2_0']['Value'],vary=params['Fit params']['Kn2_0']['Vary'],min=0) # K2 at ref temp.
    fit_params.add('dCp2', value=params['Fit params']['dCpn2']['Value'],vary=params['Fit params']['dCpn2']['Vary']) # K2 dCp
    
    fit_params.add('T0', value=params['Fit params']['T0']['Value']+273.15,vary=params['Fit params']['T0']['Vary']) # Reference temperature
    
    fit_params.add('Rh_3', value=params['Fit params']['Rh_3']['Value'], vary=params['Fit params']['Rh_3']['Vary'],min=4e-9,max=6e-9) # Trimer hydrodynamic radius
    fit_params.add('h_scale',value=params['Fit params']['h_scale']['Value'],vary=params['Fit params']['h_scale']['Vary'],min=-0.6,max=-0.1) # >6-mer Diffusion Constant Scaling
    fit_params.add('j_scale',value=params['Fit params']['j_scale']['Value'],vary=params['Fit params']['j_scale']['Vary']) # 3-6-mer Diffusion Constant Scaling
    fit_params.add('N',value=params['Fit params']['N']['Value'],vary=params['Fit params']['N']['Vary']) # Final oligomer size
    fit_params.add('k_d0',value=params['Fit params']['k_d0']['Value'],vary=params['Fit params']['k_d0']['Vary']) # Effective nonideality constant
    fit_params.add('m',value=params['Fit params']['m']['Value'],vary=params['Fit params']['m']['Vary']) # Effective nonideality constant temperature dependence
    
    fit_params.add('eta1',value=params['Eta params'][0],vary=False) # Viscosity parameters from fitting third order polynomial to SEDNTERP viscosity as a function of temperature
    fit_params.add('eta2',value=params['Eta params'][1],vary=False) # a*T^3 + b*T^2 + c*T + d, a = eta params[0], b = eta params[1], ...
    fit_params.add('eta3',value=params['Eta params'][2],vary=False)
    fit_params.add('eta4',value=params['Eta params'][3],vary=False)

    run_fit = params['Run fit'] # Run fit with 'y', simulate data with 'n'
    get_errors = params['Get errors'] # Do MC errors with 'y', bypass with 'n'
    MC_iter = params['Monte Carlo iterations'] # Number of Monte Carlo error iterations
    plot_minimization = params['Plot minimization'] # If running fit, plot minimization in real time with 'y', don't plot with 'n'
    PDFname = params['Output PDF'] # Name for fit or simulation output PDF with plots
    samplename = params['Sample name'] # Name of sample for plot text   
    figure_dir = params['Figure directory'] # Directory to save plot figures
    pop_plot_concs = params['Plot concentrations'] # Total monomer concentrations to plot histograms for
    sim_conc = params['Simulation concentration'] # Total monomer concentration to simulate oligomeric populations at once the fit is complete
    
    if run_fit == 'y': # Fit data
        
        time0 = time.time() # Calculate fit run time
        
        # Fit data    
        result = minimize(objective, fit_params, method='nelder', args=(plot_minimization, fit_groups))
        
        time1 = time.time()
        print(f"\nThe elapsed fit time is {time1-time0} s \n")
        
        report_fit(result)
        opt_params = result.params
        
        if get_errors == 'y':
            
            MC_dict, MC_errors = MonteCarloErrors(fit_data, opt_params, result, MC_iter)
        
            plot_fit(fit_groups, opt_params, PDFname, samplename, figure_dir, MC_dict, pop_plot_concs, sim_conc)
            
        if get_errors == 'n':
            
            MC_dict = []
            
            plot_fit(fit_groups, opt_params, PDFname, samplename, figure_dir, MC_dict, pop_plot_concs, sim_conc)
        
        print('\nTHERE ARE ONLY K1 AND K2 IN THIS PROGRAM!!!!!!!!!!!!!!!!!!!!!!!!')
    
        
    if run_fit == 'n': # Simulate data

        MC_dict = []
            
        plot_fit(fit_groups, fit_params, PDFname, samplename, figure_dir, MC_dict, pop_plot_concs, sim_conc)
        
        
        print('\nTHERE ARE ONLY K1 AND K2 IN THIS PROGRAM!!!!!!!!!!!!!!!!!!!!!!!!')


### Nucleated polymerization, two pathways
def supercageNucPol(fit_params, fit_group):
    
    Temperatures = np.array(fit_group.Temperature + 273.15)
    Concentrations = np.array(fit_group.Concentration*1e-6)

    m = fit_params['m'].value # Nonideality
    k_d0 = fit_params['k_d0'].value
    
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
            
            D_dict[f"D1{3*x}_0"] = [None]*len(Temperatures)
            D_dict[f"D1{3*x}"] = [None]*len(Temperatures)
            c_dict[f"c1{3*x}"] = [None]*len(Temperatures)
            P_dict[f"P1{3*x}"] = [None]*len(Temperatures)
            
            D_dict[f"D2{3*x}_0"] = [None]*len(Temperatures)
            D_dict[f"D2{3*x}"] = [None]*len(Temperatures)
            c_dict[f"c2{3*x}"] = [None]*len(Temperatures)
            P_dict[f"P2{3*x}"] = [None]*len(Temperatures)
            
        else:
            
            D_dict[f"D{3*x}_0"] = [None]*len(Temperatures)
            D_dict[f"D{3*x}"] = [None]*len(Temperatures)
            c_dict[f"c{3*x}"] = [None]*len(Temperatures)
            P_dict[f"P{3*x}"] = [None]*len(Temperatures)
            
    D_dict['Dz'] = [None]*len(Temperatures)
    
    # Calculate trimer diffusion as a function of temperature
    D3_0 = StokesEinstein(Temperatures, eta1, eta2, eta3, eta4, fit_params['Rh_3'].value) # Calculate trimer diffusion constant first
    
    # Get equilibrium constants
    K_dict = equilibriumconstants(Temperatures, fit_params)
    K1 = K_dict['K1']
    K2 = K_dict['K2']
        
    X_guess = 1e-5 # Initial guess for dimensionless trimer concentration to use in solver
    for index, Temperature in enumerate(Temperatures):
        
        ################## Solve dimensionless trimer concentration ##################
        constants = [(Concentrations[0]*K2[index]/3),K1[index]/K2[index]] # XT, alpha1 = Kn1/Kn2
        ffs_partial = partial(ffs3,constants)
        sol = opt.root(ffs_partial,X_guess,method='lm')         
        c3 = sol.x[0]/K2[index] # Trimer concentration in molar
        
        # Calculate temperature-dependent non-ideality constant
        k_d = k_d0 + m*(Temperature - fit_params['T0'].value)

        # Generate populations and concentrations from solver solutions, calculate diffusion coefficients
        D_num = 0
        D_den = 0
        for x in range(1,N+1):
            
            D3x_0 = D3_0[index]*x**(h_scale)
            
            if x == 1:
                
                c3x = c3
                c_dict[f"c{3*x}"][index] = c3x # Trimer
                P_dict[f"P{3*x}"][index] = 3*x*c3x/Concentrations[0]
                
                D_num += (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                D_den += (x**2)*c3x
                
                D_dict[f"D{3*x}_0"][index] = D3x_0
                D_dict[f"D{3*x}"][index] = D3x_0*(1 + k_d*Concentrations[0])

            if x == 2:
                    
                # Hexamer 1
                D3x_0 = D3_0[index]*x**(j_scale)
                c3x = K1[index]*c_dict[f"c{3*(x-1)}"][index]*c3
                c_dict[f"c1{3*x}"][index] = c3x # Hexamer 1 dictated by trimer and K1
                P_dict[f"P1{3*x}"][index] = 3*x*c3x/Concentrations[0] # Hexamer 1 dictated by trimer and K1
                
                D_num += (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                D_den += (x**2)*c3x
                
                D_dict[f"D1{3*x}_0"][index] = D3x_0
                D_dict[f"D1{3*x}"][index] = D3x_0*(1 + k_d*Concentrations[0])
                
                # Hexamer 2
                D3x_0 = D3_0[index]*x**(j_scale)
                c3x = K2[index]*c_dict[f"c{3*(x-1)}"][index]*c3
                c_dict[f"c2{3*x}"][index] = c3x # Hexamer 2 dictated by trimer and K2
                P_dict[f"P2{3*x}"][index] = 3*x*c3x/Concentrations[0] # Hexamer 2 dictated by trimer and K2
                
                D_num += (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                D_den += (x**2)*c3x
                
                D_dict[f"D2{3*x}_0"][index] = D3x_0
                D_dict[f"D2{3*x}"][index] = D3x_0*(1 + k_d*Concentrations[0])
                
            if x == 3:
                
                c3x = K2[index]*c_dict[f"c2{3*(x-1)}"][index]*c3
                c_dict[f"c{3*x}"][index] = c3x # 9-mer dictated by Hexamer 2 and K2
                P_dict[f"P{3*x}"][index] = 3*x*c3x/Concentrations[0]
                
                D_num += (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                D_den += (x**2)*c3x
                
                D_dict[f"D{3*x}_0"][index] = D3x_0
                D_dict[f"D{3*x}"][index] = D3x_0*(1 + k_d*Concentrations[0])
                    
            if x > 3:
                
                c3x = K2[index]*c_dict[f"c{3*(x-1)}"][index]*c3
                c_dict[f"c{3*x}"][index] = c3x # >9-mer dictated by 9-mer and K2
                P_dict[f"P{3*x}"][index] = 3*x*c3x/Concentrations[0]
                
                D_num += (x**2)*c3x*(D3x_0*(1 + k_d*Concentrations[0]))
                D_den += (x**2)*c3x
                
                D_dict[f"D{3*x}_0"][index] = D3x_0
                D_dict[f"D{3*x}"][index] = D3x_0*(1 + k_d*Concentrations[0])
        
        # Make dictionary of D values for fit, get average diffusion coefficient
        D_dict['Dz'][index] = D_num/D_den
        
        X_guess = c3*K2[index] # Solver guess for next temperature is solution at the i-1 temperature for stability
        
    return D_dict, c_dict, P_dict


# Calculate equilibrium constants
def equilibriumconstants(Temperatures, fit_params):
    
    # Set up thermodynamic and hydrodynamic parameters
    T0 = fit_params['T0'].value
    dH1_0 = fit_params['dH1_0'].value
    K1_0 = abs(fit_params['K1_0'].value)
    dCp1 = fit_params['dCp1'].value
    
    dH2_0 = fit_params['dH2_0'].value
    K2_0 = abs(fit_params['K2_0'].value)
    dCp2 = fit_params['dCp2'].value

    # Calculate equilibrium constants
    K1 = K1_0*np.exp((dH1_0/R)*((1/T0) - (1/Temperatures)) + (dCp1/R)*(np.log(Temperatures/T0) + (T0/Temperatures) - 1))
    K2 = K2_0*np.exp((dH2_0/R)*((1/T0) - (1/Temperatures)) + (dCp2/R)*(np.log(Temperatures/T0) + (T0/Temperatures) - 1))
    
    # Make dictionary of the K values
    K_dict = {'K1':[None]*len(Temperatures),'K2':[None]*len(Temperatures)}
    K_dict['K1'] = K1
    K_dict['K2'] = K2
    
    return K_dict


### Two pathways, no cooperativity, dimensionless trimer concentration solver
def ffs3(constants,X):
    
    XT, a1 = constants # Unpack constants
    
    eq1 = -XT + 2.*a1*np.square(X) + (X/np.square(X-1))
    
    return eq1


### Calculate diffusion coefficients
def StokesEinstein(Temperature, eta1, eta2, eta3, eta4, Rh):
    
    eta = eta1*np.power(Temperature,3) + eta2*np.square(Temperature) + eta3*Temperature + eta4 # Third order polynomial viscosity as a function of temperature from SEDNTERP
    Dt = (kB*Temperature)/(6*np.pi*eta*Rh)

    return Dt
    

### Minimization function for global fit
def objective(fit_params, plot_minimization, fit_groups):
    
    if plot_minimization == 'n': # Don't plot fit in real time

        resid = []
        for ind, group in fit_groups:
            
            D_dict, c_dict, P_dict = supercageNucPol(fit_params, group)            
            resid.append(np.array(group.D.values) - np.array(D_dict['Dz']))
    
    if plot_minimization == 'y': # Plot fit in real time
        
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.set_ylim(1.5e-11,8.5e-11)
        
        resid = []
        for ind, group in fit_groups:
            
            D_dict, c_dict, P_dict = supercageNucPol(fit_params, group)            
            resid.append(np.array(group.D.values) - np.array(D_dict['Dz']))
        
            ax.plot(group.Temperature,group.D.values,'ko')
            ax.plot(group.Temperature,D_dict['Dz'],'r')
    
        plt.ion()
        plt.pause(0.00001)
        plt.ioff()
        plt.close()
    
    resid = np.ravel(resid)
    
    return resid
  

### Generate errors by Monte Carlo analysis
def MonteCarloErrors(data, opt_params, fit_result, MC_iter):
    
    perfect_data = data.copy() # Make copy of dataframe
    RMSD = np.sqrt(fit_result.chisqr/fit_result.ndata)
    plot_minimization = 'n'
    
    perfect_groups = perfect_data.groupby('Concentration')
    
    for ind, group in perfect_groups: # Overwrite diffusion constants to be perfect simulated values from best fit params
    
        D_dict, c_dict, P_dict = supercageNucPol(opt_params, group)
        perfect_data.loc[perfect_data['Concentration'] == group.Concentration.iloc[0], 'D'] = D_dict['Dz']
    
    MC_dict = {k:[] for k in opt_params.keys()} # Make dictionary for Monte Carlo parameters
    errors = {k+' error':[] for k in MC_dict.keys()} # Make dictionary for errors
    counter = 1
    for x in range(MC_iter):
        
        print(f"######### The current Monte-Carlo iteration is: {counter} #########")
        
        perturbed_data = perfect_data.copy() # Copy perfect data groups
        perturbed_data.D = perturbed_data.D + np.random.normal(scale=RMSD, size=np.size(perturbed_data.D)) # Perturb perfect data for MC analysis
        perturbed_groups = perturbed_data.groupby('Concentration')

        perturbed_result = minimize(objective, opt_params, method='nelder', args=(plot_minimization, perturbed_groups))
        {MC_dict[k].append(perturbed_result.params[k]) for k in perturbed_result.params.keys()}
        
        counter = counter + 1
     
    for k in MC_dict.keys():
        errors[k+' error'].append(np.std(MC_dict[k]))
    
    for k1,k2 in zip(opt_params.keys(),errors.keys()):
        print(f"{k1} = {opt_params[k1].value} +/- {errors[k2][0]}")
        
    return MC_dict, errors


### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(fit_groups, opt_params, PDFname, samplename, figure_dir, MC_dict, pop_plot_concs, sim_conc):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(PDFname)
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.style.use('figure')
    hist_colors = ['#2166ac','#92c5de','#66c2a4','#d6604d','#b2182b',]
    hex_colors = ['#020101','#281112','#4E2128','#733243','#984364','#BD548A','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF']
    hex_colors = list(reversed(hex_colors))
    M3_color = '#fc8d59'
    M6_color = '#78c679'
    M24_color = '#2b8cbe'

    cmap = customcolormap.get_continuous_cmap(hex_colors,fit_groups.ngroups) #  Get HEX color at specified number of points using the above HEX colors
    
    if os.path.isdir(figure_dir) == False: # Check if directory for saving figures exists; if not, make it
        topdir = figure_dir.split('/')[0] + '/' + figure_dir.split('/')[1] # Figure path specified as ./Figures/Sample
        os.mkdir(topdir)
        os.mkdir(figure_dir) # Make bottom directory for the specific sample
            
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
    D_ax2.bar([1],[2.33e-7],width=0.5,color=M6_color,label='experiment') # Experiment vs HYDROPRO diffusion constants for hexamer
    D_ax2.bar([2],[2.35e-7],width=0.5,color=M6_color,label='HYDROPRO',hatch='///',edgecolor='w') 
    D_ax2.bar([3],[2.70e-7],width=0.5,color=M3_color,label='experiment') # Experiment vs HYDROPRO diffusion constants for trimer
    D_ax2.bar([4],[2.75e-7],width=0.5,color=M3_color,label='HYDROPRO',hatch='///',edgecolor='w')
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

    cidx = 0
    for ind, group in fit_groups:
        
        sim_df = pd.DataFrame()
        sim_df['Temperature'] = np.linspace(np.min(group.Temperature),np.max(group.Temperature),100)
        sim_df['Concentration'] = np.zeros(len(sim_df['Temperature']))
        sim_df['Concentration'] = group.Concentration.iloc[0]
        D_dict, c_dict, P_dict = supercageNucPol(opt_params, sim_df)

        D_ax.plot(group.Temperature,group.D*1e4,'o',color=cmap[cidx])
        D_ax.plot(sim_df['Temperature'],np.array(D_dict['Dz'])*1e4,color=cmap[cidx])
        D_ax.plot(sim_df['Temperature'],np.array(D_dict["D3_0"])*1e4,'--',color=M3_color) # 3-mer
        D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D16_0"])*1e4,'--',color=M6_color) # 6-mer
        D_ax.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*8}_0"])*1e4,'--',color=M24_color) # 24-mer

        # Plot concentration histograms
        # Temperature indices
        if ind in pop_plot_concs:
            
            Pfigs = []
            
            for x in range(len(hist_temperatures)):
                
                Pfigs.append(plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1,3]})) # Make split axes plot

                # zoom-in / limit the view to different portions of the data
                Pfigs[x][1][0].set_ylim(0.25, 1.0)  # outliers only
                Pfigs[x][1][1].set_ylim(0, 0.20)  # most of the data
                Pfigs[x][1][0].set_yticks([0.25, 1.0])

                # hide the spines between ax and ax2
                Pfigs[x][1][0].spines['bottom'].set_visible(False)
                Pfigs[x][1][1].spines['top'].set_visible(False)
                Pfigs[x][1][0].xaxis.tick_top()
                Pfigs[x][1][0].tick_params(labeltop=False)  # don't put tick labels at the top
                Pfigs[x][1][1].xaxis.tick_bottom()

                Pfigs[x][1][0].yaxis.major.formatter._useMathText = True
                Pfigs[x][1][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                Pfigs[x][1][0].set_xlim([0,41])               
                
                Pfigs[x][1][1].yaxis.major.formatter._useMathText = True
                Pfigs[x][1][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                Pfigs[x][1][1].set_xlabel("Oligomer size")
                Pfigs[x][1][1].set_xlim([0,41])
                Pfigs[x][1][1].set_ylabel('Population',color='w')
                Pfigs[x][0].text(0.01,0.4,'Population',fontsize=20,rotation='vertical')
            
            for hcindex, temperature in enumerate(hist_temperatures):

                tindex = np.argmin(np.abs(temperature - sim_df['Temperature'].values))
                
                for x in range(1,int(opt_params['N'].value)+1):
            
                    if x == 2: # For hexamer A and B
                        
                        Pfigs[hcindex][1][0].bar(3*x,P_dict[f"P1{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                        Pfigs[hcindex][1][1].bar(3*x,P_dict[f"P1{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                        Pfigs[hcindex][1][0].bar(3.5*x,P_dict[f"P2{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                        Pfigs[hcindex][1][1].bar(3.5*x,P_dict[f"P2{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                    
                    else:  # All other sizes
                        
                        Pfigs[hcindex][1][0].bar(3*x,P_dict[f"P{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                        Pfigs[hcindex][1][1].bar(3*x,P_dict[f"P{3*x}"][tindex],color=hist_colors[hcindex],edgecolor=hist_colors[hcindex],width=1)
                        
                Pfigs[hcindex][1][0].text(0.9,0.6,'{}\n$M_{{T}}$ = {} $\mu$M\n{} \N{DEGREE SIGN}C'.format(samplename,group.Concentration.iloc[0],temperature),fontsize=20,va='top',ha='right',ma='left',transform=Pfigs[hcindex][1][0].transAxes)

                Pfigs[hcindex][0].tight_layout(pad=1)
                pdf.savefig(Pfigs[hcindex][0])
                Pfigs[hcindex][0].savefig(f"{figure_dir}/Populations_{temperature}C_{group.Concentration.iloc[0]}uM.png")
                plt.close(Pfigs[hcindex][0])
            
        if ind == min(pop_plot_concs): # Lowest concentration for inset
            
            Temperatures = sim_df['Temperature'] + 273.15
            K_dict = equilibriumconstants(Temperatures, opt_params)
            
            if len(MC_dict) == 0: # No Monte Carlo errors
                
                # Just plot equilibrium constants once since they are the same in every temperature range
                K_ax.plot(1./(Temperatures),np.log(K_dict['K1']),linewidth=2,color='#4393c3',label='$K_{1}$')
                K_ax.plot(1./(Temperatures),np.log(K_dict['K2']),linewidth=2,color='#b2182b',label='$K_{2}$')
            
            if len(MC_dict) > 0: # Plot Monte Carlo confidence intervals

                confidence_Ks = {'lnK1':[],'lnK2':[]}
                    
                for index, element in enumerate(MC_dict['dH1_0']):
                        
                    dH1_0 = MC_dict['dH1_0'][index]
                    dCp1 = MC_dict['dCp1'][index]
                    K1_0 = np.abs(MC_dict['K1_0'][index])
                    dH2_0 = MC_dict['dH2_0'][index]
                    dCp2 = MC_dict['dCp2'][index]
                    K2_0 = np.abs(MC_dict['K2_0'][index])
                    T0 = MC_dict['T0'][index]
                
                    # Calculate equilibrium constants
                    K1 = K1_0*np.exp((dH1_0/R)*((1/T0) - (1/Temperatures)) + (dCp1/R)*(np.log(Temperatures/T0) + (T0/Temperatures) - 1))
                    K2 = K2_0*np.exp((dH2_0/R)*((1/T0) - (1/Temperatures)) + (dCp2/R)*(np.log(Temperatures/T0) + (T0/Temperatures) - 1))
                
                    confidence_Ks['lnK1'].append(np.log(K1))
                    confidence_Ks['lnK2'].append(np.log(K2))
            
                    lnK1_upper = np.mean(confidence_Ks['lnK1'],0) + 1.96*np.std(confidence_Ks['lnK1'],0)
                    lnK1_lower = np.mean(confidence_Ks['lnK1'],0) - 1.96*np.std(confidence_Ks['lnK1'],0)
                    lnK2_upper = np.mean(confidence_Ks['lnK2'],0) + 1.96*np.std(confidence_Ks['lnK2'],0)
                    lnK2_lower = np.mean(confidence_Ks['lnK2'],0) - 1.96*np.std(confidence_Ks['lnK2'],0)
            
                # Just plot equilibrium constants once since they are the same in every temperature range
                K_ax.plot(1./(Temperatures),np.log(K_dict['K1']),linewidth=2,color='#4393c3',label='$K_{1}$')
                K_ax.fill_between(1./(Temperatures),lnK1_lower,lnK1_upper,facecolor='#4393c3',edgecolor='#4393c3',alpha=0.1)
                K_ax.plot(1./(Temperatures),np.log(K_dict['K2']),linewidth=2,color='#b2182b',label='$K_{2}$')
                K_ax.fill_between(1./(Temperatures),lnK2_lower,lnK2_upper,facecolor='#b2182b',edgecolor='#b2182b',alpha=0.1)
                                  
                with open(f"{figure_dir}/lnK_confidence_dict.p", 'wb') as fp: # Write to dictionary
                    
                    confidence_Ks['Temperature'] = Temperatures
                    save_dict = confidence_Ks
                    pickle.dump(save_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
                      
            Kax_leg = K_ax.legend(loc='upper left',frameon=False,handlelength=0)

            for line, text in zip(Kax_leg.get_lines(), Kax_leg.get_texts()):
                text.set_color(line.get_color())
            
            # Diffusion constant inset at high temperature
            # Multiplying by 1e11 to make scale on inset match the full plot
            D_axins.plot(group.Temperature,group.D*1e11,'o',markersize=6,color=cmap[cidx])
            D_axins.plot(sim_df['Temperature'],np.array(D_dict['Dz'])*1e11,linewidth=2,color=cmap[cidx])
            D_axins.plot(sim_df['Temperature'],np.array(D_dict["D3_0"])*1e11,'--',linewidth=2,color=M3_color) # 3-mer
            D_axins.plot(sim_df['Temperature'],np.array(D_dict[f"D16_0"])*1e11,'--',linewidth=2,color=M6_color) # 6-mer
            D_axins.plot(sim_df['Temperature'],np.array(D_dict[f"D{3*8}_0"])*1e11,'--',linewidth=2,color=M24_color) # 24-mer
            D_axins.set_xticks([30, 40, 50])
            D_axins.set_yticks([5,6,7,8])
            D_axins.set_xlim([29.5,50.5])
            D_axins.set_ylim([4.8,8.5])
            D_axins.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
            D_axins.yaxis.major.formatter._useMathText = True
            
        cidx += 1
    
    K_fig.tight_layout(pad=1)
    pdf.savefig(K_fig)
    if len(MC_dict) == 0:
        K_fig.savefig(f"{figure_dir}/VantHoffplot.png",format='png')
    if len(MC_dict) > 0:
        K_fig.savefig(f"{figure_dir}/VantHoffplot_confidenceintervals.png",format='png')
    plt.close(K_fig)
    
    D_fig.tight_layout(pad=1)
    D_fig2.tight_layout(pad=1)
    pdf.savefig(D_fig)
    pdf.savefig(D_fig2)
    D_fig.savefig(f"{figure_dir}/full_Dz.png",format='png')
    D_fig2.savefig(f"{figure_dir}/Dbar.png",format='png')
    plt.close(D_fig)
    plt.close(D_fig2)
    pdf.close()
    
    # Simulate populations and concentrations at desired biological monomer concentrations specified in parameter file
    for concentration in sim_conc:
    
        sim_df = pd.DataFrame()
        sim_df['Temperature'] = np.linspace(5,50,100)
        sim_df['Concentration'] = np.zeros(len(sim_df['Temperature']))
        sim_df['Concentration'] = concentration # total monomer concentration in uM to simulate oligomeric distribution at
        D_dict, c_dict, P_dict = supercageNucPol(opt_params, sim_df)
    
        with open(f"{figure_dir}/Populations_{sim_df['Concentration'].iloc[0]}uM.p", 'wb') as fp:
            save_dict = P_dict
            save_dict['Temperature'] = sim_df['Temperature']
            pickle.dump(save_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()
            
        with open(f"{figure_dir}/Concentrations_{sim_df['Concentration'].iloc[0]}uM.p", 'wb') as fp:
            save_dict = c_dict
            save_dict['Temperature'] = sim_df['Temperature']
            pickle.dump(save_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()


R = 1.9872e-3 # Gas constant for calculating affinities
R = R*4.184*1000 # Thermodynamic parameters have to be in J/mol and J/mol*K to match this gas constant
kB = 1.38065e-23 # Boltzmann constant for viscosity
main()