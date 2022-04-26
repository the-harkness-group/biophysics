#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:39:44 2021

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
        fit_params.add(f"w_{label}_directmax",value=group['w2_ppm'].iloc[-1]+(group['w2_ppm'].iloc[0]-group['w2_ppm'].iloc[-1]),min=0)
        fit_params.add(f"w_{label}_directmin",value=group['w2_ppm'].iloc[-1],min=0,vary=False)
        fit_params.add(f"w_{label}_indirectmax",value=group['w1_ppm'].iloc[-1]+(group['w1_ppm'].iloc[0]-group['w1_ppm'].iloc[-1]),min=0)
        fit_params.add(f"w_{label}_indirectmin",value=group['w1_ppm'].iloc[-1],min=0,vary=False)

        fit_params.add(f"Kd_{label}_direct",value=expt_params['Kd'],min=0) # Per-residue Kd
        fit_params.add(f"Kd_{label}_indirect",value=expt_params['Kd'],min=0)
            
    model = expt_params['model']

    result = minimize(objective, fit_params, args=(expt_params,expt_data,model)) # run global fit
    print(fit_report(result)) # print fit report
    
    plot_fit(expt_params, expt_data, result.params, model) # plot experiment and fitted data
    
    # Write .csv files with fit parameters
    opt_params_dict = {f"{k}":[result.params[f"{k}"].value] for k in result.params.keys()}
    opt_params_df = pd.DataFrame(opt_params_dict)
    opt_params_df.to_csv(f"{expt_params['Fit_PDF'].split('.pdf')[0]}_optimalparameters.csv")


### Minimization function for global fit
def objective(fit_params, expt_params, expt_data, model):

    groups = expt_data.groupby('Assignment',sort=False)
    
    resid = []
            
    if model == 'P + L <-> PL':

        for ind, group in groups:
            
            PT = np.array(expt_params['PT'])*1e-6
            LT = np.array(group['Concentration_uM'])*1e-6
            
            assignment = ind.split('-')[0] + ind.split('-')[1]
            dw_B = fit_params[f"w_{assignment}_directmax"].value
            dw_F = fit_params[f"w_{assignment}_directmin"].value
            Kd = fit_params[f"Kd_{assignment}_direct"].value
            resid.append(group['w2_ppm'].values - twostateinter(Kd, PT, LT, dw_B, dw_F))
            
            dw_B = fit_params[f"w_{assignment}_indirectmax"].value
            dw_F = fit_params[f"w_{assignment}_indirectmin"].value
            Kd = fit_params[f"Kd_{assignment}_indirect"].value
            resid.append(group['w1_ppm'].values - twostateinter(Kd, PT, LT, dw_B, dw_F))
    
    resid = np.ravel(resid)
    
    return resid


### Two-state fast-exchange intermolecular
def twostateinter(Kd, PT, LT, dw_B, dw_F):
    
    a = (1/Kd)
    b = (1/Kd)*PT - (1/Kd)*LT + 1
    c = -LT
    
    L = (-b + np.sqrt(b**2 - 4.*a*c))/(2.*a)
    PL = LT - L
    
    PB = PL/PT
    PF = 1 - PB
    
    CSP = PB*dw_B + PF*dw_F  # chemical shift perturbation, per residue dw

    return CSP


### Plot optimized fit result and save pdfs of plots for each peak
### Can handle two-state intra and two-state inter binding at the moment
def plot_fit(expt_params, expt_data, result_params, model):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(expt_params['Fit_PDF'])
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.style.use('figure')
    colors = expt_params['Bar_color']

    groups = expt_data.groupby('Assignment',sort=False)
    
    for ind, group in groups:
        
        assignment = ind.split('-')[0] + ind.split('-')[1]

        fig, ax = plt.subplots(1,2,figsize=(15,7))
        ax[0].set_title(group.Assignment.iloc[0])
        ax[1].set_title(group.Assignment.iloc[0])

        if model == 'P + L <-> PL':
            LT_sim = np.linspace(min(np.array(group['Concentration_uM']))*1e-6,max(np.array(group['Concentration_uM']))*1e-6,100)
            LT = np.array(group['Concentration_uM'])*1e-6
            PT = np.array(expt_params['PT'])*1e-6
            
            Kd = result_params[f"Kd_{assignment}_direct"].value
            dw_B = result_params[f"w_{assignment}_directmax"].value
            dw_F = result_params[f"w_{assignment}_directmin"].value
            opt_CSP_direct = twostateinter(Kd, PT, LT_sim, dw_B, dw_F) 
            
            Kd = result_params[f"Kd_{assignment}_indirect"].value
            dw_B = result_params[f"w_{assignment}_indirectmax"].value
            dw_F = result_params[f"w_{assignment}_indirectmin"].value
            opt_CSP_indirect = twostateinter(Kd, PT, LT_sim, dw_B, dw_F)
            
            ax[0].plot(LT*1e3,group['w2_ppm'],'o',color=colors)
            ax[0].plot(LT_sim*1e3,opt_CSP_direct,color=colors,linewidth=2)
            ax[0].set_xlabel('$L_{T}$ mM')
            ax[0].set_ylabel('Direct ppm')
            ax[0].text(0.45,0.55,"$K_{D}$ = %.2e $\pm$ %.2e M\n$\omega_{B}$ = %.2f $\pm$ %.2f ppm"
              %(result_params[f"Kd_{assignment}_direct"],result_params[f"Kd_{assignment}_direct"].stderr,
                result_params[f"w_{assignment}_directmax"].value,result_params[f"w_{assignment}_directmax"].stderr),
                transform=ax[0].transAxes,va="top")
            
            ax[1].plot(LT*1e3,group['w1_ppm'],'o',color=colors)
            ax[1].plot(LT_sim*1e3,opt_CSP_indirect,color=colors,linewidth=2)
            ax[1].set_xlabel('$L_{T}$ mM')
            ax[1].set_ylabel('Indirect ppm')
            ax[1].text(0.45,0.55,"$K_{D}$ = %.2e $\pm$ %.2e M\n$\omega_{B}$ = %.2f $\pm$ %.2f ppm"
              %(result_params[f"Kd_{assignment}_indirect"],result_params[f"Kd_{assignment}_indirect"].stderr,
                result_params[f"w_{assignment}_indirectmax"].value,result_params[f"w_{assignment}_indirectmax"].stderr),
                transform=ax[1].transAxes,va="top")
            
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    pdf.close()

### Run fit and generate result plots
result = main()