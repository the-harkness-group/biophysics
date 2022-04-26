#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 15:32:51 2021

@author: robertharkness
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
import yaml
from lmfit import minimize, Parameters, report_fit
from scipy import optimize as opt
from functools import partial
import os
import pickle
import copy

### Read in data and set up for fitting
def main():
    
    params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(params['Dataset']) # Get dataset name to fit
    data = data[data['Concentration'] < 800]
    groups = data.groupby('Temperature')
        
    fit_params = Parameters() # Set up fit/simulation parameters

    fit_params.add('dH1o', value=params['Fit params']['dHn1_0']['Value'],vary=params['Fit params']['dHn1_0']['Vary']) # dH1 at ref temp.
    fit_params.add('K1o', value=params['Fit params']['Kn1_0']['Value'],vary=params['Fit params']['Kn1_0']['Vary'],min=0) # K1 at ref temp.
    fit_params.add('dCp1', value=params['Fit params']['dCpn1']['Value'],vary=params['Fit params']['dCpn1']['Vary']) # K1 dCp
    
    fit_constants = {'To':params['Fit params']['T0']['Value']+273.15,'S3_0':params['Fit params']['S3_0']['Value'],
                     'j_scale':params['Fit params']['j_scale']['Value'],'R':8.3145e-3}
    
    MC_iter = params['Monte Carlo iterations'] # Number of Monte-Carlo error estimate iterations
    plot_minimization = params['Plot minimization'] # If running fit, plot minimization in real time with 'y', don't plot with 'n'
    
    # Fit data
    result = minimize(objective, fit_params, method='nelder', args=(plot_minimization, groups, fit_constants))
    report_fit(result)
    opt_params = copy.deepcopy(result.params)
    
    print(fit_params)
    print(opt_params)
    
    # Get errors
    MC_dict, MC_data, errors = MonteCarloErrors(data, opt_params, result, MC_iter, fit_constants)
    
    # Plot
    plot_fit(groups, opt_params, errors, MC_dict, MC_data, params, fit_constants)


### Nucleated polymerization, two pathways
def supercageNucPol(fit_params, fit_group, fit_constants):
    
    Temperatures = np.array(fit_group.Temperature + 273.15)
    Concentrations = np.array(fit_group.Concentration*1e-6)
    
    j_scale = fit_constants['j_scale']
    
    # Set up dictionaries for species up to size N
    S_dict = {}
    c_dict = {}
    P_dict = {}
            
    S_dict[f"S3_0"] = [None]*len(Concentrations)
    c_dict[f"c3"] = [None]*len(Concentrations)
    P_dict[f"P3"] = [None]*len(Concentrations)
    
    S_dict[f"S6_0"] = [None]*len(Concentrations)
    c_dict[f"c6"] = [None]*len(Concentrations)
    P_dict[f"P6"] = [None]*len(Concentrations)
            
    S_dict['Sw'] = [None]*len(Concentrations)
    
    S3_0 = fit_constants['S3_0']
    
    # Get equilibrium constants
    K_dict = equilibriumconstants(Temperatures, fit_params, fit_constants)
    K1 = K_dict['K1']
        
    X_guess = [0.5,0.5]
    for index, Concentration in enumerate(Concentrations):
        
        ################## Solve dimensionless trimer concentration ##################
        constants = [Concentration, K1[index]] # XT, alpha1 = Kn1/Kn2
        ffs_partial = partial(ffs3,constants)
        sol = opt.root(ffs_partial,X_guess,method='lm')
        P3 = sol.x[0]
        P6 = sol.x[1]

        # Generate populations and concentrations from solver solutions
        c_dict['c3'][index] = P3*Concentration/3
        c_dict['c6'][index] = P6*Concentration/6
        P_dict['P3'][index] = P3
        P_dict['P6'][index] = P6
        S_dict['S3_0'][index] = S3_0
        S_dict['S6_0'][index] = S3_0*2**(j_scale)
        S_dict['Sw'][index] = S_dict['S3_0'][index]*P_dict['P3'][index] + S_dict['S6_0'][index]*P_dict['P6'][index]
    
    return S_dict, c_dict, P_dict


# Calculate equilibrium constants
def equilibriumconstants(Temperatures, fit_params, fit_constants):
    
    # Set up thermodynamic and hydrodynamic parameters
    T0 = fit_constants['To']
    dH1_0 = fit_params['dH1o'].value
    K1_0 = abs(fit_params['K1o'].value)
    dCp1 = fit_params['dCp1'].value
    R = fit_constants['R']

    # Calculate equilibrium constants
    K1 = K1_0*np.exp((dH1_0/R)*((1/T0) - (1/Temperatures)) + (dCp1/R)*(np.log(Temperatures/T0) + (T0/Temperatures) - 1))

    # Make dictionary of the K values
    K_dict = {}
    K_dict['K1'] = K1
    
    return K_dict


### Two pathways, no cooperativity, dimensionless trimer concentration solver
def ffs3(constants,p):
    
    CT, K1 = constants # Unpack constants
    P3, P6 = p
    
    M3 = P3*CT/3
    M6 = P6*CT/6
    
    eq1 = -1 + P3 + P6
    eq2 = K1*M3**2 - M6
    
    return [eq1, eq2]


### Minimization function for global fit
def objective(fit_params, plot_minimization, fit_groups, fit_constants):

    resid = []
    for ind, group in fit_groups:
            
        S_dict, c_dict, P_dict = supercageNucPol(fit_params, group, fit_constants)
        resid.append(np.array(group.Sw.values) - np.array(S_dict['Sw']))
        #resid.append((np.array(group.Sw.values) - np.array(S_dict['Sw']))/np.array(group.Sw.values))
    
    resid = np.hstack(resid)
    
    return resid


### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(groups, opt_params, errors, MC_dict, MC_data, params, fit_constants):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DegP2_AUC_isothermfits.pdf')
    plt.style.use('figure')
    colors = ['#4575b4','#91bfdb','#66c2a4']
    hist_colors = colors
    figure_dir = params['Figure directory']
    samplename = params['Sample name']
    
    if os.path.isdir(figure_dir) == False: # Check if directory for saving figures exists; if not, make it
        topdir = figure_dir.split('/')[0] + '/' + figure_dir.split('/')[1] # Figure path specified as ./Figures/Sample
        os.mkdir(topdir)
        os.mkdir(figure_dir) # Make bottom directory for the specific sample
           
    S_fig, ax1 = plt.subplots(1,1)
    ax1.set_xscale('log')
    ax1.set_xlabel("$\it{M_{T}}$ $\mu$M")
    ax1.set_ylabel('$\it{s_{w,0}}$ S')

    # Make dummy dataframe to get s-value limits
    sim_df = pd.DataFrame()
    sim_df['Concentration'] = np.linspace(2.1e-6,171e-6,100)
    sim_df['Temperature'] = np.zeros(len(sim_df['Concentration']))
    sim_df['Temperature'] = 298.15
    S_dict, c_dict, P_dict = supercageNucPol(opt_params, sim_df, fit_constants)  
    ax1.tick_params(axis='x',pad=10)
    ax1.plot(sim_df['Concentration']*1e6,S_dict['S3_0'],'--',linewidth=2,color='#fc8d59')
    ax1.plot(sim_df['Concentration']*1e6,S_dict['S6_0'],'--',linewidth=2,color='#78c679')
    ax1.text(0.10,0.97,"$M_{6}$",color='#78c679',fontsize=36,transform=ax1.transAxes,va="top")
    ax1.text(0.82,0.275,"$M_{3}$",color='#fc8d59',fontsize=36,transform=ax1.transAxes,va="top")
    ax1.text(0.10,0.71,"8 \N{DEGREE SIGN}C",color='#4575b4',fontsize=24,transform=ax1.transAxes,va="top")
    ax1.text(0.10,0.55,"20 \N{DEGREE SIGN}C",color='#91bfdb',fontsize=24,transform=ax1.transAxes,va="top")
    ax1.text(0.10,0.23,"30 \N{DEGREE SIGN}C",color='#66c2a4',fontsize=24,transform=ax1.transAxes,va="top")
    ax1.set_ylim([5.5,11.7])
    ax1.set_xlim([1.8,200])
    
    K_fig = plt.figure()
    K_ax = K_fig.add_subplot(111)
    K_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
    K_ax.yaxis.major.formatter._useMathText = True
    K_ax.xaxis.major.formatter._useMathText = True
    K_ax.set_xlabel("1/T $K^{-1}$")
    K_ax.set_ylabel('ln(K)')
    K_ax.set_ylim([1.2e1,1.8e1])
    
    Temperatures = []
    for color_idx, (ind, group) in enumerate(groups):
        
        sim_df = pd.DataFrame()
        sim_df['Concentration'] = np.linspace(np.min(group.Concentration),np.max(group.Concentration),100)
        sim_df['Temperature'] = np.zeros(len(sim_df['Concentration']))
        sim_df['Temperature'] = group.Temperature.iloc[0]
        S_dict, c_dict, P_dict = supercageNucPol(opt_params, sim_df, fit_constants)
        
        ax1.plot(group.Concentration,group.Sw,'o',color=colors[color_idx])
        ax1.plot(sim_df['Concentration'],S_dict['Sw'],linewidth=2,color=colors[color_idx])
        ax1.fill_between(sim_df['Concentration'],MC_data[ind]['Lower bound'],MC_data[ind]['Upper bound'],facecolor=colors[color_idx],edgecolor=colors[color_idx],alpha=0.1)
        
        Pfigs = [] # Plot concentration histograms
        
        Temperatures.append(group.Temperature.iloc[0]) # Get temperatures for calculating Van 't Hoff plot
            
        for x in range(len(group.Concentration)):
                
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
            Pfigs[x][1][0].set_xlim([0,41])               
                
            Pfigs[x][1][1].yaxis.major.formatter._useMathText = True
            Pfigs[x][1][1].set_xlabel("Oligomer size")
            Pfigs[x][1][1].set_xlim([0,41])
            Pfigs[x][1][1].set_ylabel('Population',color='w')
            Pfigs[x][0].text(0.01,0.33,'Population',fontsize=36,rotation='vertical')
            
        for hcindex, concentration in enumerate(group.Concentration):

            tindex = np.argmin(np.abs(concentration - sim_df['Concentration'].values))
            
            Pfigs[hcindex][1][0].bar(3,P_dict[f"P3"][tindex],color=hist_colors[color_idx],edgecolor=hist_colors[color_idx],width=1)
            Pfigs[hcindex][1][1].bar(3,P_dict[f"P3"][tindex],color=hist_colors[color_idx],edgecolor=hist_colors[color_idx],width=1)
                        
            Pfigs[hcindex][1][0].bar(6,P_dict[f"P6"][tindex],color=hist_colors[color_idx],edgecolor=hist_colors[color_idx],width=1)
            Pfigs[hcindex][1][1].bar(6,P_dict[f"P6"][tindex],color=hist_colors[color_idx],edgecolor=hist_colors[color_idx],width=1)
                            
            Pfigs[hcindex][1][0].text(0.9,0.6,'{}\n$M_{{T}}$ = {} $\mu$M\n{} \N{DEGREE SIGN}C'.format(samplename,concentration,group.Temperature.iloc[0]),fontsize=20,va='top',ha='right',ma='left',transform=Pfigs[hcindex][1][0].transAxes)

            Pfigs[hcindex][0].tight_layout(pad=1)
            pdf.savefig(Pfigs[hcindex][0])
            Pfigs[hcindex][0].savefig(f"{figure_dir}/Populations_{group.Temperature.iloc[0]}C_{concentration}uM.pdf",format='pdf')
            plt.close(Pfigs[hcindex][0])
        
    sim_Temps = np.linspace(np.min(Temperatures),np.max(Temperatures),100) + 273.15
    K_dict = equilibriumconstants(sim_Temps, opt_params, fit_constants)
    R = fit_constants['R']
            
    if len(MC_dict) == 0: # No Monte Carlo errors
                
        # Just plot equilibrium constants once since they are the same in every temperature range
        K_ax.plot(1./(sim_Temps),np.log(K_dict['K1']),linewidth=2,color='#4393c3',label='$K_{1}$')
        K_ax.plot(1./(sim_Temps),np.log(K_dict['K2']),linewidth=2,color='#b2182b',label='$K_{2}$')
            
    if len(MC_dict) > 0: # Plot Monte Carlo confidence intervals

        confidence_Ks = {'lnK1':[],'lnK2':[]}
                    
        for index, element in enumerate(MC_dict['dH1o']):
                        
            dH1_0 = MC_dict['dH1o'][index]
            dCp1 = MC_dict['dCp1'][index]
            K1_0 = np.abs(MC_dict['K1o'][index])
            T0 = fit_constants['To']
                
            # Calculate equilibrium constants
            K1 = K1_0*np.exp((dH1_0/R)*((1/T0) - (1/sim_Temps)) + (dCp1/R)*(np.log(sim_Temps/T0) + (T0/sim_Temps) - 1))
  
            confidence_Ks['lnK1'].append(np.log(K1))
            
            lnK1_upper = np.mean(confidence_Ks['lnK1'],0) + 1.96*np.std(confidence_Ks['lnK1'],0)
            lnK1_lower = np.mean(confidence_Ks['lnK1'],0) - 1.96*np.std(confidence_Ks['lnK1'],0)

        # Just plot equilibrium constants once since they are the same in every temperature range
        K_ax.plot(1./(sim_Temps),np.log(K_dict['K1']),linewidth=2,color='#4393c3',label='$K_{1}$')
        K_ax.fill_between(1./(sim_Temps),lnK1_lower,lnK1_upper,facecolor='#4393c3',edgecolor='#4393c3',alpha=0.1)
      
        Kax_leg = K_ax.legend(loc='upper left',frameon=False,handlelength=0)

        for line, text in zip(Kax_leg.get_lines(), Kax_leg.get_texts()):
            text.set_color(line.get_color())
    
    K_fig.tight_layout(pad=1)
    pdf.savefig(K_fig)
    if len(MC_dict) == 0:
        K_fig.savefig(f"{figure_dir}/VantHoffplot.pdf",format='pdf')
    if len(MC_dict) > 0:
        K_fig.savefig(f"{figure_dir}/VantHoffplot_confidenceintervals.pdf",format='pdf')
    plt.close(K_fig)
    
    with open(f"{figure_dir}/lnK_confidence_dict.p", 'wb') as fp: # Write to dictionary
        confidence_Ks['Temperature'] = sim_Temps - 273.15
        pickle.dump(confidence_Ks, fp, protocol=pickle.HIGHEST_PROTOCOL)

    
    S_fig.tight_layout(pad=1)
    pdf.savefig(S_fig)
    S_fig.savefig(f"{figure_dir}/Sw.pdf",format='pdf')
    plt.close(S_fig)
    pdf.close()
 
    
### Generate errors by Monte Carlo analysis
def MonteCarloErrors(data, opt_params, fit_result, MC_iter, fit_constants):
    
    perfect_data = data.copy() # Make copy of dataframe
    RMSD = np.sqrt(fit_result.chisqr/fit_result.ndata)
    plot_minimization = 'n'
    
    perfect_groups = perfect_data.groupby('Temperature')
    
    for ind, group in perfect_groups: # Overwrite diffusion constants to be perfect simulated values from best fit params
    
        S_dict, c_dict, P_dict = supercageNucPol(opt_params, group, fit_constants)
        perfect_data.loc[perfect_data['Temperature'] == group.Temperature.iloc[0], 'Sw'] = S_dict['Sw']
    
    MC_dict = {k:[] for k in opt_params.keys()} # Make dictionary for Monte Carlo parameters
    errors = {k+' error':[] for k in MC_dict.keys()} # Make dictionary for errors
    counter = 1
    for x in range(MC_iter):
        
        print(f"######### The current Monte-Carlo iteration is: {counter} #########")
        
        perturbed_data = perfect_data.copy() # Copy perfect data groups
        perturbed_data.Sw = perturbed_data.Sw + np.random.normal(scale=RMSD, size=np.size(perturbed_data.Sw)) # Perturb perfect data for MC analysis
        perturbed_groups = perturbed_data.groupby('Temperature')

        perturbed_result = minimize(objective, opt_params, method='nelder', args=(plot_minimization, perturbed_groups, fit_constants))
        {MC_dict[k].append(perturbed_result.params[k]) for k in perturbed_result.params.keys()}
        
        counter = counter + 1
     
    for k in MC_dict.keys():
        errors[k+' error'].append(np.std(MC_dict[k]))
    
    for k1,k2 in zip(opt_params.keys(),errors.keys()):
        print(f"{k1} = {opt_params[k1].value} +/- {errors[k2][0]}")
        
    with open('MC_parameter_dictionary.pickle','wb') as f:
        pickle.dump(MC_dict,f,protocol=pickle.HIGHEST_PROTOCOL)
    
    # Make simulated curves for plotting confidence intervals
    MC_data = {}
    perfect_groups = perfect_data.groupby('Temperature')
    for ind, group in perfect_groups: # Overwrite CSP values to be perfect values from optimal fit
        
        MC_data[ind] = {'Simulated data':[],'Upper bound':[],'Lower bound':[]}
        
        sim_df = pd.DataFrame()
        sim_df['Concentration'] = np.linspace(np.min(group.Concentration),np.max(group.Concentration),100)
        sim_df['Temperature'] = np.zeros(len(sim_df['Concentration']))
        sim_df['Temperature'] = group.Temperature.iloc[0]
        
        for x in range(MC_iter):
            
            MCsimparams = copy.deepcopy(opt_params)
            
            for k in MC_dict:
                
                if k in opt_params:
            
                    MCsimparams[k].value = MC_dict[k][x]
            
            S_dict, c_dict, P_dict = supercageNucPol(MCsimparams, sim_df, fit_constants)
            MC_data[ind]['Simulated data'].append(S_dict['Sw']) # Store data simulated from Monte Carlo parameters
        
        MC_data[ind]['Upper bound'] = np.mean(MC_data[ind]['Simulated data'],0) + 1.96*np.std(MC_data[ind]['Simulated data'],0) # Calculate upper and lower 95% confidence intervals
        MC_data[ind]['Lower bound'] = np.mean(MC_data[ind]['Simulated data'],0) - 1.96*np.std(MC_data[ind]['Simulated data'],0)
    
    return MC_dict, MC_data, errors

main()