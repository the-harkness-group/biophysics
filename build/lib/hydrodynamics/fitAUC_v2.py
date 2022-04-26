#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:09:18 2021

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

### Read in data and set up for fitting
def main():
    
    params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(params['Dataset']) # Get dataset name to fit
    #data = data[data['Concentration'] < 800]
    groups = data.groupby('Temperature')
        
    fit_params = Parameters() # Set up fit/simulation parameters

    fit_params.add('dH1_0', value=params['Fit params']['dHn1_0']['Value'],vary=params['Fit params']['dHn1_0']['Vary']) # dH1 at ref temp.
    fit_params.add('K1_0', value=params['Fit params']['Kn1_0']['Value'],vary=params['Fit params']['Kn1_0']['Vary'],min=0) # K1 at ref temp.
    fit_params.add('dCp1', value=params['Fit params']['dCpn1']['Value'],vary=params['Fit params']['dCpn1']['Vary']) # K1 dCp
    
    fit_params.add('dH2_0', value=params['Fit params']['dHn2_0']['Value'],vary=params['Fit params']['dHn2_0']['Vary']) # dH2 at ref temp.
    fit_params.add('K2_0', value=params['Fit params']['Kn2_0']['Value'],vary=params['Fit params']['Kn2_0']['Vary'],min=0) # K2 at ref temp.
    fit_params.add('dCp2', value=params['Fit params']['dCpn2']['Value'],vary=params['Fit params']['dCpn2']['Vary']) # K2 dCp
    
    fit_params.add('T0', value=params['Fit params']['T0']['Value']+273.15,vary=params['Fit params']['T0']['Vary']) # Reference temperature
    fit_params.add('R',value=8.3145e-3,vary=False)
    
    fit_params.add('S3_0', value=params['Fit params']['S3_0']['Value'], vary=params['Fit params']['S3_0']['Vary'],min=4.5,max=7) # Trimer hydrodynamic radius
    fit_params.add('h_scale',value=params['Fit params']['h_scale']['Value'],vary=params['Fit params']['h_scale']['Vary'],min=0.4,max=0.9) # >6-mer Diffusion Constant Scaling
    fit_params.add('j_scale',value=params['Fit params']['j_scale']['Value'],vary=params['Fit params']['j_scale']['Vary'],min=0.4,max=0.9) # 3-6-mer Diffusion Constant Scaling
    fit_params.add('N',value=params['Fit params']['N']['Value'],vary=params['Fit params']['N']['Vary']) # Final oligomer size
    
    MC_iter = params['Monte Carlo iterations'] # Number of Monte-Carlo error estimate iterations
    plot_minimization = params['Plot minimization'] # If running fit, plot minimization in real time with 'y', don't plot with 'n'
    
    # Fit data
    result = minimize(objective, fit_params, method='nelder', args=(plot_minimization, groups))
    report_fit(result)
    opt_params = result.params
    
    # Get errors
    MC_dict, MC_data, errors = MonteCarloErrors(data, opt_params, result, MC_iter)
    
    # Plot
    plot_fit(groups, opt_params, errors, MC_dict, MC_data, params)


### Nucleated polymerization, two pathways
def supercageNucPol(fit_params, fit_group):
    
    Temperatures = np.array(fit_group.Temperature + 273.15)
    Concentrations = np.array(fit_group.Concentration*1e-6)

    N = int(fit_params['N'].value)
    h_scale = fit_params['h_scale'].value
    j_scale = fit_params['j_scale'].value
    
    # Set up dictionaries for species up to size N
    S_dict = {}
    c_dict = {}
    P_dict = {}
    for x in range(1,N+1):
        
        if x == 2:
            
            S_dict[f"S1{3*x}_0"] = [None]*len(Concentrations)
            c_dict[f"c1{3*x}"] = [None]*len(Concentrations)
            P_dict[f"P1{3*x}"] = [None]*len(Concentrations)
            
            S_dict[f"S2{3*x}_0"] = [None]*len(Concentrations)
            c_dict[f"c2{3*x}"] = [None]*len(Concentrations)
            P_dict[f"P2{3*x}"] = [None]*len(Concentrations)
            
        else:
            
            S_dict[f"S{3*x}_0"] = [None]*len(Concentrations)
            c_dict[f"c{3*x}"] = [None]*len(Concentrations)
            P_dict[f"P{3*x}"] = [None]*len(Concentrations)
            
    S_dict['Sw'] = [None]*len(Concentrations)
    
    S3_0 = fit_params['S3_0'].value
    
    # Get equilibrium constants
    K_dict = equilibriumconstants(Temperatures, fit_params)
    K1 = K_dict['K1']
    K2 = K_dict['K2']
        
    X_guess = 1e-3 # Initial guess for dimensionless trimer concentration to use in solver
    for index, Concentration in enumerate(Concentrations):
        
        ################## Solve dimensionless trimer concentration ##################
        constants = [(Concentration*K2[index]/3),K1[index]/K2[index]] # XT, alpha1 = Kn1/Kn2
        ffs_partial = partial(ffs3,constants)
        sol = opt.root(ffs_partial,X_guess,method='lm')         
        c3 = sol.x[0]/K2[index] # Trimer concentration in molar

        # Generate populations and concentrations from solver solutions, calculate diffusion coefficients
        S_dict['Sw'][index] = 0
        for x in range(1,N+1):
            
            S3x_0 = S3_0*x**(h_scale)
            
            if x == 1:
                
                c3x = c3
                c_dict[f"c{3*x}"][index] = c3x # Trimer
                P_dict[f"P{3*x}"][index] = 3*x*c3x/Concentration
            
                S_dict[f"S{3*x}_0"][index] = S3x_0
                S_dict['Sw'][index] += S3x_0*P_dict[f"P{3*x}"][index]

            if x == 2:
                    
                # Hexamer 1
                S3x_0 = S3_0*x**(j_scale)
                c3x = K1[index]*c_dict[f"c{3*(x-1)}"][index]*c3
                c_dict[f"c1{3*x}"][index] = c3x # Hexamer 1 dictated by trimer and K1
                P_dict[f"P1{3*x}"][index] = 3*x*c3x/Concentration # Hexamer 1 dictated by trimer and K1
                
                S_dict[f"S1{3*x}_0"][index] = S3x_0
                S_dict['Sw'][index] += S3x_0*P_dict[f"P1{3*x}"][index]
                
                # Hexamer 2
                S3x_0 = S3_0*x**(j_scale)
                c3x = K2[index]*c_dict[f"c{3*(x-1)}"][index]*c3
                c_dict[f"c2{3*x}"][index] = c3x # Hexamer 2 dictated by trimer and K2
                P_dict[f"P2{3*x}"][index] = 3*x*c3x/Concentration # Hexamer 2 dictated by trimer and K2
                
                S_dict[f"S2{3*x}_0"][index] = S3x_0
                S_dict['Sw'][index] += S3x_0*P_dict[f"P2{3*x}"][index]
                
            if x == 3:
                
                c3x = K2[index]*c_dict[f"c2{3*(x-1)}"][index]*c3
                c_dict[f"c{3*x}"][index] = c3x # 9-mer dictated by Hexamer 2 and K2
                P_dict[f"P{3*x}"][index] = 3*x*c3x/Concentration
                
                S_dict[f"S{3*x}_0"][index] = S3x_0
                S_dict['Sw'][index] += S3x_0*P_dict[f"P{3*x}"][index]
                    
            if x > 3:
                
                c3x = K2[index]*c_dict[f"c{3*(x-1)}"][index]*c3
                c_dict[f"c{3*x}"][index] = c3x # >9-mer dictated by 9-mer and K2
                P_dict[f"P{3*x}"][index] = 3*x*c3x/Concentration
                
                S_dict[f"S{3*x}_0"][index] = S3x_0
                S_dict['Sw'][index] += S3x_0*P_dict[f"P{3*x}"][index]
        
        #X_guess = c3*K2[index] # Solver guess for next temperature is solution at the i-1 temperature for stability
    
    return S_dict, c_dict, P_dict


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
    
    R = fit_params['R'].value

    # Calculate equilibrium constants
    K1 = K1_0*np.exp((dH1_0/R)*((1/T0) - (1/Temperatures)) + (dCp1/R)*(np.log(Temperatures/T0) + (T0/Temperatures) - 1))
    K2 = K2_0*np.exp((dH2_0/R)*((1/T0) - (1/Temperatures)) + (dCp2/R)*(np.log(Temperatures/T0) + (T0/Temperatures) - 1))
    
    # Make dictionary of the K values
    K_dict = {'K1':[],'K2':[]}
    K_dict['K1'] = K1
    K_dict['K2'] = K2
    
    return K_dict


### Two pathways, no cooperativity, dimensionless trimer concentration solver
def ffs3(constants,X):
    
    XT, a1 = constants # Unpack constants
    
    eq1 = -XT + 2.*a1*np.square(X) + (X/np.square(X-1))
    
    return eq1


### Minimization function for global fit
def objective(fit_params, plot_minimization, fit_groups):
    
    if plot_minimization == 'n': # Don't plot fit in real time

        resid = []
        for ind, group in fit_groups:
            
            S_dict, c_dict, P_dict = supercageNucPol(fit_params, group)
            resid.append(np.array(group.Sw.values) - np.array(S_dict['Sw']))
    
    if plot_minimization == 'y': # Plot fit in real time
        
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        #ax.set_ylim(1.5e-11,8.5e-11)
        
        resid = []
        for ind, group in fit_groups:
            
            S_dict, c_dict, P_dict = supercageNucPol(fit_params, group)            
            resid.append(np.array(group.Sw.values) - np.array(S_dict['Sw']))
        
            ax.plot(np.log(group.Concentration),group.Sw.values,'ko')
            ax.plot(np.log(group.Concentration),S_dict['Sw'],'r')
    
        plt.ion()
        plt.pause(0.00001)
        plt.ioff()
        plt.close()
    
    resid = np.hstack(resid)
    
    return resid


### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(groups, opt_params, errors, MC_dict, MC_data, params):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DegP2_AUC_isothermfits.pdf')
    plt.style.use('figure')
    colors = ['#4575b4','#91bfdb','#66c2a4']
    hist_colors = colors
    #hist_colors = ['#2166ac','#92c5de','#66c2a4','#d6604d','#b2182b']
    figure_dir = params['Figure directory']
    samplename = params['Sample name']
    
    if os.path.isdir(figure_dir) == False: # Check if directory for saving figures exists; if not, make it
        topdir = figure_dir.split('/')[0] + '/' + figure_dir.split('/')[1] # Figure path specified as ./Figures/Sample
        os.mkdir(topdir)
        os.mkdir(figure_dir) # Make bottom directory for the specific sample
           
    S_fig, ax1 = plt.subplots(1,1)
    ax1.set_xscale('log')
    ax1.set_xlabel("$M_{T}$ $\mu$M")
    ax1.set_ylabel('$s_{w}$ S')

    # Make dummy dataframe to get s-value limits
    sim_df = pd.DataFrame()
    sim_df['Concentration'] = np.linspace(2.1e-6,896e-6,100)
    #sim_df['Concentration'] = np.linspace(2.1e-6,171e-6,100)
    sim_df['Temperature'] = np.zeros(len(sim_df['Concentration']))
    sim_df['Temperature'] = 298.15
    S_dict, c_dict, P_dict = supercageNucPol(opt_params, sim_df)  
    ax1.tick_params(axis='x',pad=10)
    ax1.plot(sim_df['Concentration']*1e6,S_dict['S3_0'],'--',linewidth=2,color='#fc8d59')
    ax1.plot(sim_df['Concentration']*1e6,S_dict['S16_0'],'--',linewidth=2,color='#78c679')
    #ax1.plot(sim_df['Concentration']*1e6,S_dict['S12_0'],'--',linewidth=2,color='#d53e4f')
    #ax1.text(0.85,0.9,"$M_{12}$",color='#d53e4f',fontsize=24,transform=ax1.transAxes,va="top")
    #ax1.text(0.85,0.40,"$M_{6}$",color='#99d594',fontsize=24,transform=ax1.transAxes,va="top")
    #ax1.text(0.85,0.2,"$M_{3}$",color='#3288bd',fontsize=24,transform=ax1.transAxes,va="top")
    ax1.text(0.10,0.95,"$M_{6}$",color='#78c679',fontsize=24,transform=ax1.transAxes,va="top")
    ax1.text(0.82,0.23,"$M_{3}$",color='#fc8d59',fontsize=24,transform=ax1.transAxes,va="top")
    ax1.text(0.10,0.70,"8 \N{DEGREE SIGN}C",color='#4575b4',fontsize=24,transform=ax1.transAxes,va="top")
    ax1.text(0.10,0.55,"20 \N{DEGREE SIGN}C",color='#91bfdb',fontsize=24,transform=ax1.transAxes,va="top")
    ax1.text(0.10,0.20,"30 \N{DEGREE SIGN}C",color='#66c2a4',fontsize=24,transform=ax1.transAxes,va="top")
    #ax1.set_xlim([1.8,1050])
    #ax1.set_ylim([5.5,11.5])
    #ax1.set_xlim([1.8,200])
    
    K_fig = plt.figure()
    K_ax = K_fig.add_subplot(111)
    K_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
    K_ax.yaxis.major.formatter._useMathText = True
    K_ax.xaxis.major.formatter._useMathText = True
    K_ax.set_xlabel("1/T $K^{-1}$")
    K_ax.set_ylabel('ln(K)')
    
    Temperatures = []
    for color_idx, (ind, group) in enumerate(groups):
        
        sim_df = pd.DataFrame()
        sim_df['Concentration'] = np.linspace(np.min(group.Concentration),np.max(group.Concentration),100)
        sim_df['Temperature'] = np.zeros(len(sim_df['Concentration']))
        sim_df['Temperature'] = group.Temperature.iloc[0]
        S_dict, c_dict, P_dict = supercageNucPol(opt_params, sim_df)  

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
                
            for x in range(1,int(opt_params['N'].value)+1):
            
                if x == 2: # For hexamer A and B
                        
                    Pfigs[hcindex][1][0].bar(3*x,P_dict[f"P1{3*x}"][tindex],color=hist_colors[color_idx],edgecolor=hist_colors[color_idx],width=1)
                    Pfigs[hcindex][1][1].bar(3*x,P_dict[f"P1{3*x}"][tindex],color=hist_colors[color_idx],edgecolor=hist_colors[color_idx],width=1)
                    Pfigs[hcindex][1][0].bar(3.5*x,P_dict[f"P2{3*x}"][tindex],color=hist_colors[color_idx],edgecolor=hist_colors[color_idx],width=1)
                    Pfigs[hcindex][1][1].bar(3.5*x,P_dict[f"P2{3*x}"][tindex],color=hist_colors[color_idx],edgecolor=hist_colors[color_idx],width=1)
                    
                else:  # All other sizes
                        
                    Pfigs[hcindex][1][0].bar(3*x,P_dict[f"P{3*x}"][tindex],color=hist_colors[color_idx],edgecolor=hist_colors[color_idx],width=1)
                    Pfigs[hcindex][1][1].bar(3*x,P_dict[f"P{3*x}"][tindex],color=hist_colors[color_idx],edgecolor=hist_colors[color_idx],width=1)
                        
            Pfigs[hcindex][1][0].text(0.9,0.6,'{}\n$M_{{T}}$ = {} $\mu$M\n{} \N{DEGREE SIGN}C'.format(samplename,concentration,group.Temperature.iloc[0]),fontsize=20,va='top',ha='right',ma='left',transform=Pfigs[hcindex][1][0].transAxes)

            Pfigs[hcindex][0].tight_layout(pad=1)
            pdf.savefig(Pfigs[hcindex][0])
            Pfigs[hcindex][0].savefig(f"{figure_dir}/Populations_{group.Temperature.iloc[0]}C_{concentration}uM.png")
            plt.close(Pfigs[hcindex][0])
        
    sim_Temps = np.linspace(np.min(Temperatures),np.max(Temperatures),100) + 273.15
    K_dict = equilibriumconstants(sim_Temps, opt_params)
            
    if len(MC_dict) == 0: # No Monte Carlo errors
                
        # Just plot equilibrium constants once since they are the same in every temperature range
        K_ax.plot(1./(sim_Temps),np.log(K_dict['K1']),linewidth=2,color='#4393c3',label='$K_{1}$')
        K_ax.plot(1./(sim_Temps),np.log(K_dict['K2']),linewidth=2,color='#b2182b',label='$K_{2}$')
            
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
            R = MC_dict['R'][index]
                
            # Calculate equilibrium constants
            K1 = K1_0*np.exp((dH1_0/R)*((1/T0) - (1/sim_Temps)) + (dCp1/R)*(np.log(sim_Temps/T0) + (T0/sim_Temps) - 1))
            K2 = K2_0*np.exp((dH2_0/R)*((1/T0) - (1/sim_Temps)) + (dCp2/R)*(np.log(sim_Temps/T0) + (T0/sim_Temps) - 1))
                
            confidence_Ks['lnK1'].append(np.log(K1))
            confidence_Ks['lnK2'].append(np.log(K2))
            
            lnK1_upper = np.mean(confidence_Ks['lnK1'],0) + 1.96*np.std(confidence_Ks['lnK1'],0)
            lnK1_lower = np.mean(confidence_Ks['lnK1'],0) - 1.96*np.std(confidence_Ks['lnK1'],0)
            lnK2_upper = np.mean(confidence_Ks['lnK2'],0) + 1.96*np.std(confidence_Ks['lnK2'],0)
            lnK2_lower = np.mean(confidence_Ks['lnK2'],0) - 1.96*np.std(confidence_Ks['lnK2'],0)
            
        # Just plot equilibrium constants once since they are the same in every temperature range
        K_ax.plot(1./(sim_Temps),np.log(K_dict['K1']),linewidth=2,color='#4393c3',label='$K_{1}$')
        K_ax.fill_between(1./(sim_Temps),lnK1_lower,lnK1_upper,facecolor='#4393c3',edgecolor='#4393c3',alpha=0.1)
        K_ax.plot(1./(sim_Temps),np.log(K_dict['K2']),linewidth=2,color='#b2182b',label='$K_{2}$')
        K_ax.fill_between(1./(sim_Temps),lnK2_lower,lnK2_upper,facecolor='#b2182b',edgecolor='#b2182b',alpha=0.1)
                      
        Kax_leg = K_ax.legend(loc='upper left',frameon=False,handlelength=0)

        for line, text in zip(Kax_leg.get_lines(), Kax_leg.get_texts()):
            text.set_color(line.get_color())
    
    K_fig.tight_layout(pad=1)
    pdf.savefig(K_fig)
    if len(MC_dict) == 0:
        K_fig.savefig(f"{figure_dir}/VantHoffplot.png",format='png')
    if len(MC_dict) > 0:
        K_fig.savefig(f"{figure_dir}/VantHoffplot_confidenceintervals.png",format='png')
    plt.close(K_fig)
    
    with open(f"{figure_dir}/lnK_confidence_dict.p", 'wb') as fp: # Write to dictionary
        confidence_Ks['Temperature'] = sim_Temps - 273.15
        pickle.dump(confidence_Ks, fp, protocol=pickle.HIGHEST_PROTOCOL)

    
    S_fig.tight_layout(pad=1)
    pdf.savefig(S_fig)
    S_fig.savefig(f"{figure_dir}/Sw.png",format='png')
    plt.close(S_fig)
    pdf.close()
 
    
### Generate errors by Monte Carlo analysis
def MonteCarloErrors(data, opt_params, fit_result, MC_iter):
    
    perfect_data = data.copy() # Make copy of dataframe
    RMSD = np.sqrt(fit_result.chisqr/fit_result.ndata)
    plot_minimization = 'n'
    
    perfect_groups = perfect_data.groupby('Temperature')
    
    for ind, group in perfect_groups: # Overwrite diffusion constants to be perfect simulated values from best fit params
    
        S_dict, c_dict, P_dict = supercageNucPol(opt_params, group)
        perfect_data.loc[perfect_data['Temperature'] == group.Temperature.iloc[0], 'Sw'] = S_dict['Sw']
    
    MC_dict = {k:[] for k in opt_params.keys()} # Make dictionary for Monte Carlo parameters
    errors = {k+' error':[] for k in MC_dict.keys()} # Make dictionary for errors
    counter = 1
    for x in range(MC_iter):
        
        print(f"######### The current Monte-Carlo iteration is: {counter} #########")
        
        perturbed_data = perfect_data.copy() # Copy perfect data groups
        perturbed_data.Sw = perturbed_data.Sw + np.random.normal(scale=RMSD, size=np.size(perturbed_data.Sw)) # Perturb perfect data for MC analysis
        perturbed_groups = perturbed_data.groupby('Temperature')

        perturbed_result = minimize(objective, opt_params, method='nelder', args=(plot_minimization, perturbed_groups))
        {MC_dict[k].append(perturbed_result.params[k]) for k in perturbed_result.params.keys()}
        
        counter = counter + 1
     
    for k in MC_dict.keys():
        errors[k+' error'].append(np.std(MC_dict[k]))
    
    for k1,k2 in zip(opt_params.keys(),errors.keys()):
        print(f"{k1} = {opt_params[k1].value} +/- {errors[k2][0]}")
         
    # Make simulated curves for plotting confidence intervals
    MC_data = {}
    perfect_groups = perfect_data.groupby('Temperature')
    for ind, group in perfect_groups: # Overwrite CSP values to be perfect values from optimal fit
        
        MC_data[ind] = {'Simulated data':[],'Upper bound':[],'Lower bound':[]}
        
        sim_df = pd.DataFrame()
        sim_df['Concentration'] = np.linspace(np.min(group.Concentration),np.max(group.Concentration),100)
        #sim_df['Concentration'] = np.linspace(np.min(group.Concentration),896,1000)
        sim_df['Temperature'] = np.zeros(len(sim_df['Concentration']))
        sim_df['Temperature'] = group.Temperature.iloc[0]
        
        for x in range(MC_iter):
            
            MCsimparams = opt_params
            
            for k in MC_dict:
                
                if k in opt_params:
            
                    MCsimparams[k].value = MC_dict[k][x]
            
            S_dict, c_dict, P_dict = supercageNucPol(MCsimparams, sim_df)
            MC_data[ind]['Simulated data'].append(S_dict['Sw']) # Store data simulated from Monte Carlo parameters
        
        MC_data[ind]['Upper bound'] = np.mean(MC_data[ind]['Simulated data'],0) + 1.96*np.std(MC_data[ind]['Simulated data'],0) # Calculate upper and lower 95% confidence intervals
        MC_data[ind]['Lower bound'] = np.mean(MC_data[ind]['Simulated data'],0) - 1.96*np.std(MC_data[ind]['Simulated data'],0)
    
    return MC_dict, MC_data, errors

main()