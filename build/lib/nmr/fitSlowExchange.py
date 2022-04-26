#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:56:27 2019

@author: robertharkness
"""

import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
from lmfit import minimize, Parameters, report_fit

### Read in data and set up for fitting
def main():

    # Set up parameters
    params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(params['Dataset'])

    fit_params = Parameters()
    groups = data.groupby('Assignment')
    for ind, group in groups:
        fit_params.add(f"factor_{ind}",value=max(data[data['Assignment'] == ind].Intensity.values),vary=0)
        fit_params.add(f"intercept_{ind}",value=0)
    fit_params.add('Kd',value=1e-6)

    # Run fit
    result = minimize(objective, fit_params, args=(data,params))
    report_fit(result)
    
    # Plot fit result
    plot_fit(data, groups, result, params)

### Minimization function for global fit
def objective(fit_params, data, params):

    resid = [] # Residuals
    for x in range(len(data)):
        Kd = fit_params['Kd'].value
        assignment = data.Assignment.iloc[x]
        factor = fit_params[f"factor_{assignment}"].value
        intercept = fit_params[f"intercept_{assignment}"].value
        I = TwoStateSlowExchange(Kd, factor, intercept, data.State.iloc[x],
         data.Concentration.iloc[x], data.Receptor.iloc[x])
        resid.append(data.Intensity.iloc[x] - I)

    return resid

### Two-state slow exchange NMR ligand binding
def TwoStateSlowExchange(Kd, factor, intercept, state, ligand, receptor):

    ligand = ligand*1e-6
    receptor = receptor*1e-6
    r = ligand/receptor
    c = receptor/Kd
    PB = 0.5*(1 + r + (1/c) - np.sqrt((1 + r + (1/c))**2 - 4*r))
    
    if state == 'Free': # Free peak
        I = factor*(1 - PB) + intercept

    if state == 'Bound': # Bound peak
        I = factor*PB + intercept

    return I

### Plot optimized fit result and save pdfs of plots for each peak
def plot_fit(data, groups, result, params):
    
    plt.style.use('figure')
    pdf = matplotlib.backends.backend_pdf.PdfPages('slowexchangefits.pdf')
    colors = [plt.cm.plasma(y) for y in range(150)]

    conc_sim = np.linspace(min(data.Concentration),max(data.Concentration),100)
    for ind, group in groups:

        Kd = result.params['Kd'].value
        factor = result.params[f"factor_{ind}"].value
        intercept = result.params[f"intercept_{ind}"].value

        receptor_sim = np.array([group.Receptor.iloc[0] for x in range(len(conc_sim))])
        opt_fit = TwoStateSlowExchange(Kd, factor, intercept, group.State.iloc[0], conc_sim, receptor_sim)

        fig,ax = plt.subplots(1,1,figsize=(6,4.8))
        #ax.errorbar(LT/PT,group.amp,yerr=group.amp_err,fmt="o",capsize=5,elinewidth=2,markeredgewidth=2,color=colors[145])
        ax.plot(group.Concentration.values,group.Intensity.values,'o',color=colors[145])
        ax.plot(conc_sim,opt_fit,color=colors[145],linewidth=2)
        ax.set_xlabel('$L_{T}$ $\mu$M')
        ax.set_ylabel('Peak intensity')
        ax.set_title(f"{ind} {group.State.iloc[0]}")
 
 #       if 'B' in ind:
 #           ax.text(0.5,0.45,"$I_{B}= \lambda P_{B}+c$\n$\lambda$ = %.1e $\pm$ %.1e\nc = %.1e $\pm$ %.1e\n$K_{D}$ = %3.1d $\pm$ %1.1d $\mu$M"%(result.params[f"factor_{x}"].value,result.params[f"factor_{x}"].stderr,result.params[f"intercept_{x}"].value,result.params[f"intercept_{x}"].stderr,result.params['Kd'].value*1e6,result.params['Kd'].stderr*1e6),transform=ax.transAxes,va="top")
 #           
 #       elif 'F' in ind:
 #           ax.text(0.5,0.45,"$I_{F}= \lambda (1-P_{B})+c$\n$\lambda$ = %.1e $\pm$ %.1e\nc = %.1e $\pm$ %.1e\n$K_{D}$ = %3.1d $\pm$ %1.1d $\mu$M"%(result.params[f"factor_{x}"].value,result.params[f"factor_{x}"].stderr,result.params[f"intercept_{x}"].value,result.params[f"intercept_{x}"].stderr,result.params['Kd'].value*1e6,result.params['Kd'].stderr*1e6),transform=ax.transAxes,va="top")
        fig.subplots_adjust(left=0.05)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
    pdf.close()
    
### Run fit and generate result plots
main()



