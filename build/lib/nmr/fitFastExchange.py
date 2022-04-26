#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:04:10 2020

@author: robertharkness
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
import yaml
from lmfit import minimize, Parameters, fit_report

### Read in data and set up for fitting
def main():
    
    expt_params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(expt_params['Fit_data'])
    expt_data = data[data['Assignment'].isin(expt_params['Fit_residues'])]

    ### Set up parameters
    fit_params = Parameters()
    groups = expt_data.groupby('Assignment',sort=False)
    for ind, group in groups:
        
        label = ind.split('-')[0] + ind.split('-')[1]
        fit_params.add(f"dw_{label}",value=expt_params['delta_omega'],min=0)

        if expt_params['Global Kd'] == 'n':
            
            fit_params.add(f"Kd_{label}",value=expt_params['Kd'],min=0) # Per-residue Kd
            
    if expt_params['Global Kd'] == 'y':
        
        fit_params.add('Kd',value=expt_params['Kd']) # Single global Kd
        
    model = expt_params['model']

    result = minimize(objective, fit_params, args=(expt_params,expt_data,model)) # run global fit
    print(fit_report(result)) # print fit report
    
    ### Monte Carlo errors
    # Do Monte Carlo error analysis if Kd is shared globally between residues, take error as standard deviation of Kds for each residue if on per-residue basis
        
    MC_iter = expt_params['Monte Carlo iterations']
    MC_dict, errors, MC_data = MonteCarloErrors(expt_data, result.params, expt_params, model, result, MC_iter)
    
    plot_fit(expt_params, expt_data, result.params, model, MC_data, errors) # plot experiment and fitted data
    
    # Make dataframe of experimental, best-fit, and fraction bound profiles
    # Assumes individual Kds per residue
    data_groups = expt_data.groupby('Assignment',sort=False)
    expt_dict = {'Assignment':[],'dw_expt':[],'dw_fit':[],'fb_expt':[],'fb_fit':[],
                   'LT':[],'PT':[],'Model':[]}
    result_dict = {'Assignment':[],'dw_fit':[],'fb_fit':[],
                   'LT':[],'PT':[],'Model':[]}
    for ind, group in data_groups:
        
        PT = np.array(expt_params['PT'])*1e-6
        LT = np.array(group['Concentration_uM'])*1e-6
        assignment = ind.split('-')[0] + ind.split('-')[1]
        dw = result.params[f"dw_{assignment}"].value
        Kd = result.params[f"Kd_{assignment}"].value
        
        for index, value in enumerate(LT):
            expt_dict['Assignment'].append(ind)
            expt_dict['dw_expt'].append(group.CSP_Hz.iloc[index])
            expt_dict['dw_fit'].append(twostateinter(Kd, PT, LT[index], dw))
            expt_dict['fb_expt'].append(group.CSP_Hz.iloc[index]/dw)
            expt_dict['fb_fit'].append(twostateinter(Kd, PT, LT[index], dw)/dw)
            expt_dict['LT'].append(LT[index]*1e6)
            expt_dict['PT'].append(PT*1e6)
            expt_dict['Model'].append(expt_params['model'])
        
        LT_sim = np.linspace(min(np.array(group['Concentration_uM']))*1e-6,max(np.array(group['Concentration_uM']))*1e-6,100)
        for index, value in enumerate(LT_sim):
            result_dict['Assignment'].append(ind)
            result_dict['dw_fit'].append(twostateinter(Kd, PT, LT_sim[index], dw))
            result_dict['fb_fit'].append(twostateinter(Kd, PT, LT_sim[index], dw)/dw)
            result_dict['LT'].append(LT_sim[index]*1e6)
            result_dict['PT'].append(PT*1e6)
            result_dict['Model'].append(expt_params['model'])
    
    # Write .csv files with fits
    opt_params_dict = {f"{k}":[result.params[f"{k}"].value] for k in result.params.keys()}
    opt_params_df = pd.DataFrame(opt_params_dict)
    expt_df = pd.DataFrame(expt_dict)
    result_df = pd.DataFrame(result_dict)
    opt_params_df.to_csv(f"{expt_params['Fit_PDF'].split('.pdf')[0]}_optimalparameters.csv")
    expt_df.to_csv(f"{expt_params['Fit_PDF'].split('.pdf')[0]}.csv")
    result_df.to_csv(f"{expt_params['Fit_PDF'].split('.pdf')[0]}_smooth.csv")

### Minimization function for global fit
def objective(fit_params, expt_params, expt_data, model):

    groups = expt_data.groupby('Assignment',sort=False)
    
    resid = []
    
    if expt_params['Global Kd'] == 'y': # Global Kd
        
        Kd = abs(fit_params['Kd'].value)
    
    if model == '2P <-> P2':
        
        for ind, group in groups:
            
            PT = np.array(group['Concentration_uM'])*1e-6
            
            assignment = ind.split('N-H')[0][1:]
            dw = fit_params[f"dw_{assignment}"].value
            
            if expt_params['Global Kd'] == 'n':
                Kd = fit_params[f"Kd_{assignment}"].value

            if expt_params['Unit'] == 'Hz':
                resid.append(group.CSP_Hz.values - twostateintra(Kd, PT, dw))
            
            if expt_params['Unit'] == 'ppm':
                resid.append(group.CSP_ppm.values - twostateintra(Kd, PT, dw))

            
    if model == 'P + L <-> PL':

        for ind, group in groups:
            
            PT = np.array(expt_params['PT'])*1e-6
            LT = np.array(group['Concentration_uM'])*1e-6
            
            assignment = ind.split('-')[0] + ind.split('-')[1]
            dw = fit_params[f"dw_{assignment}"].value
            
            if expt_params['Global Kd'] == 'n':
                Kd = fit_params[f"Kd_{assignment}"].value
            
            if expt_params['Unit'] == 'Hz':
                resid.append(group.CSP_Hz.values - twostateinter(Kd, PT, LT, dw))
                
            if expt_params['Unit'] == 'ppm':
                resid.append(group.CSP_ppm.values - twostateinter(Kd, PT, LT, dw))
    
    return resid


### Two-state fast-exchange self-association
def twostateintra(Kd, PT, dw):
    
    P = (-1 + np.sqrt(1. + 8.*(1./Kd)*PT))/(4.*(1./Kd)) # monomer
    P2 = (PT - P)/2. # dimer
    
    CSP = (P2/PT)*dw # chemical shift perturbation, per residue dw

    return CSP


### Two-state fast-exchange intermolecular
def twostateinter(Kd, PT, LT, dw):
    
    a = (1/Kd)
    b = (1/Kd)*PT - (1/Kd)*LT + 1
    c = -LT
    
    L = (-b + np.sqrt(b**2 - 4.*a*c))/(2.*a)
    PL = LT - L
    
    CSP = (PL/PT)*dw # chemical shift perturbation, per residue dw

    return CSP


### Generate errors by Monte Carlo analysis
def MonteCarloErrors(data, opt_params, expt_params, model, fit_result, MC_iter):
    
    perfect_data = data.copy() # Make copy of dataframe
    RMSD = np.sqrt(fit_result.chisqr/fit_result.ndata)
    
    if expt_params['Global Kd'] == 'y':
        Kd = abs(opt_params['Kd'].value)
        
    perfect_groups = perfect_data.groupby('Assignment',sort=False)
    PT = np.array(expt_params['PT'])*1e-6 # Protein concentration
    
    for ind, group in perfect_groups: # Overwrite CSP values to be perfect values from optimal fit
        
        LT = np.array(group['Concentration_uM'])*1e-6 # Params needed for fitting
        assignment = ind.split('-')[0] + ind.split('-')[1]
        dw = opt_params[f"dw_{assignment}"].value
        
        if expt_params['Global Kd'] == 'n':
            Kd = opt_params[f"Kd_{assignment}"].value
        
        perfect_data.loc[perfect_data['Assignment'] == ind, 'CSP_Hz'] = twostateinter(Kd, PT, LT, dw)

    
    MC_dict = {}
    MC_dict = {k:[] for k in opt_params.keys()} # Make dictionary for Monte Carlo parameters
    errors = {k+' error':[] for k in MC_dict.keys()} # Make dictionary for errors
    counter = 1
    for x in range(MC_iter):
        
        print(f"######### The current Monte-Carlo iteration is: {counter} #########")
        
        perturbed_data = perfect_data.copy() # Copy perfect simulated data
        perturbed_data['CSP_Hz'] = perturbed_data['CSP_Hz'] + np.random.normal(scale=RMSD, size=np.size(perturbed_data.CSP_Hz)) # Perturb perfect data for MC analysis

        perturbed_result = minimize(objective, opt_params, args=(expt_params,perturbed_data,model))
        
        {MC_dict[k].append(perturbed_result.params[k]) for k in perturbed_result.params.keys()}
        
        counter = counter + 1
     
    for k in MC_dict.keys():
        errors[k+' error'].append(np.std(MC_dict[k]))
    
    for k1,k2 in zip(opt_params.keys(),errors.keys()):
        print(f"{k1} = {opt_params[k1].value} +/- {errors[k2][0]}")
        
    # Make simulated curves for plotting confidence intervals
    MC_data = {}
    for ind, group in perfect_groups: # Overwrite CSP values to be perfect values from optimal fit
        
        MC_data[ind] = {'Simulated data':[],'Upper bound':[],'Lower bound':[]}
        
        for x in range(MC_iter):
            
            LT_sim = np.linspace(min(np.array(group['Concentration_uM']))*1e-6,max(np.array(group['Concentration_uM']))*1e-6,100)
            assignment = ind.split('-')[0] + ind.split('-')[1]
            dw = MC_dict[f"dw_{assignment}"][x].value
            
            if expt_params['Global Kd'] == 'y':
                Kd = MC_dict['Kd'][x].value
                
            if expt_params['Global Kd'] == 'n':
                Kd = MC_dict[f"Kd_{assignment}"][x].value
                
            MC_data[ind]['Simulated data'].append(twostateinter(Kd, PT, LT_sim, dw))
        
        MC_data[ind]['Upper bound'] = np.mean(MC_data[ind]['Simulated data'],0) + 1.96*np.std(MC_data[ind]['Simulated data'],0) # Calculate upper and lower 95% confidence intervals
        MC_data[ind]['Lower bound'] = np.mean(MC_data[ind]['Simulated data'],0) - 1.96*np.std(MC_data[ind]['Simulated data'],0)
        
    if expt_params['Global Kd'] == 'n':
        
        Kd_list = []
        for k in opt_params.keys():
            if k.split('_')[0] == 'Kd':
                Kd_list.append(opt_params[k].value)
            
        print(f"\nError for per-residue Kds: {np.mean(Kd_list)*1e6} +/- {np.std(Kd_list)*1e6}\n")
        errors = {}
        errors['Mean Kd'] = np.mean(Kd_list)*1e6
        errors['Kd std'] = np.std(Kd_list)*1e6
        
    return MC_dict, errors, MC_data


### Plot optimized fit result and save pdfs of plots for each peak
### Can handle two-state intra and two-state inter binding at the moment
def plot_fit(expt_params, expt_data, result_params, model, MC_data, errors):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(expt_params['Fit_PDF'])
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.style.use('figure')
    colors = expt_params['Bar_color']
    overlay_color = expt_params['Overlay color']
    
    groups = expt_data.groupby('Assignment',sort=False)
    
    if expt_params['Global Kd'] == 'y':
        Kd = result_params['Kd'].value
        Kd_error = result_params['Kd'].value
    
    if expt_params['Overlay'] == 'y':
        overlay_fig = plt.figure()
        overlay_ax = overlay_fig.add_subplot(111)
        overlay_ax.set_title(f"{expt_params['Type']} {expt_params['Temperature']} \N{DEGREE SIGN}C",color=overlay_color)
        if model == '2P <-> P2':
            overlay_ax.set_xlabel('$[M]_{T}$ mM')
            overlay_ax.set_ylabel('CSP Hz')
        if model == 'P + L <-> PL':
            overlay_ax.set_xlabel('$L_{T}$ mM')
            overlay_ax.set_ylabel('CSP Hz')
    
    for ind, group in groups:
        assignment = ind.split('-')[0] + ind.split('-')[1]
        dw = result_params[f"dw_{assignment}"].value
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(group.Assignment.iloc[0])

        if model == '2P <-> P2':
            PT_sim = np.linspace(min(np.array(group['Concentration_uM']))*1e-6,max(np.array(group['Concentration_uM']))*1e-6,100)
            PT = np.array(group['Concentration_uM'])*1e-6
            
            if expt_params['Global Kd'] == 'n':
                Kd = result_params[f"Kd_{assignment}"].value
                Kd_error = result_params[f"Kd_{assignment}"].stderr
                
            opt_CSP = twostateintra(Kd, PT_sim, dw)
            
            ax.plot(PT*1e3,group.CSP_Hz,'o',color=colors)
            ax.plot(PT_sim*1e3,opt_CSP,color=colors,linewidth=2)
            ax.set_xlabel('$[M]_{T}$ mM')
            ax.set_ylabel('CSP Hz')
            ax.text(0.8,0.1,"$K_{D}$ = %.2e $\pm$ %.2e M\n$\Delta \omega$ = %.2f $\pm$ %.2f Hz"%(Kd,Kd_error,result_params[f"dw_{assignment}"].value,result_params[f"dw_{assignment}"].stderr),transform=ax.transAxes,va="top")

        if model == 'P + L <-> PL':
            LT_sim = np.linspace(min(np.array(group['Concentration_uM']))*1e-6,max(np.array(group['Concentration_uM']))*1e-6,100)
            LT = np.array(group['Concentration_uM'])*1e-6
            PT = np.array(expt_params['PT'])*1e-6
            
            if expt_params['Global Kd'] == 'n':
                Kd = result_params[f"Kd_{assignment}"].value
                Kd_error = result_params[f"Kd_{assignment}"].stderr
                
            opt_CSP = twostateinter(Kd, PT, LT_sim, dw) 
            
            ax.plot(LT*1e3,group.CSP_Hz,'o',color=colors)
            ax.plot(LT_sim*1e3,opt_CSP,color=colors,linewidth=2)
            ax.set_xlabel('$L_{T}$ mM')
            ax.set_ylabel('CSP Hz')
            ax.text(0.55,0.25,"$K_{D}$ = %.2e $\pm$ %.2e M\n$\Delta \omega$ = %.2f $\pm$ %.2f Hz"%(Kd,Kd_error,result_params[f"dw_{assignment}"].value,result_params[f"dw_{assignment}"].stderr),transform=ax.transAxes,va="top")
            
            if ind in expt_params['Overlay residues']:
                overlay_ax.plot(LT*1e3,group.CSP_Hz,'o',color=overlay_color)
                overlay_ax.plot(LT_sim*1e3,opt_CSP,color=overlay_color,linewidth=2)
                overlay_ax.fill_between(LT_sim*1e3,MC_data[ind]['Lower bound'],MC_data[ind]['Upper bound'],facecolor=overlay_color,edgecolor=overlay_color,alpha=0.3)
                overlay_ax.set_xticks([0,1,2,3,4,5])
                offset_ind = expt_params['Overlay residues'].index(ind)
                if expt_params['Type'] == 'Amides':
                    overlay_ax.annotate(ind.split('N-H')[0],xy=(LT[2]*1e3,group.CSP_Hz.values[2]),xycoords='data',xytext=(LT[2]*1e3,group.CSP_Hz.values[2]+expt_params['Overlay offsets'][offset_ind]),textcoords='data',color=overlay_color,fontsize=24)
                
                if expt_params['Type'] == 'Methyls':
                    overlay_ax.annotate(ind.split('C')[0],xy=(LT[2]*1e3,group.CSP_Hz.values[2]),xycoords='data',xytext=(LT[2]*1e3,group.CSP_Hz.values[2]+expt_params['Overlay offsets'][offset_ind]),textcoords='data',color=overlay_color,fontsize=24)
        
        plt.subplots_adjust(bottom=0.2,left=0.2)
        pdf.savefig(fig)
        plt.close(fig)
    
    if expt_params['Global Kd'] == 'y':
        overlay_ax.text(0.25,0.13,"$K_{D}$ = %.1d $\pm$ %.1d $\mu$M"%(np.round(result_params['Kd'].value*1e6),np.round(errors['Kd error'][0]*1e6)),fontsize=24,color=overlay_color,transform=overlay_ax.transAxes,va="top")
    
    if expt_params['Global Kd'] == 'n':
        overlay_ax.text(0.25,0.13,"$K_{D}$ = %.1d $\pm$ %.1d $\mu$M"%(np.round(errors['Mean Kd']),np.round(errors['Kd std'])),fontsize=24,color=overlay_color,transform=overlay_ax.transAxes,va="top")
    
    overlay_fig.tight_layout()
    overlay_fig.savefig('overlay_fits.pdf',format='pdf')
    pdf.savefig(overlay_fig)
    plt.close(overlay_fig)
    
    pdf.close()

### Run fit and generate result plots
result = main()