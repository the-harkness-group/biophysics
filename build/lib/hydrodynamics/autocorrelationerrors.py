#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:41:50 2021

@author: robertharkness
"""

import sys
import yaml
import pandas as pd
from scipy.constants import pi
import numpy as np
from lmfit import Model
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pickle

def main():
    
############################################################################
##################### SET UP AND FIT #######################################
############################################################################

    ### Get list of wells from yaml file to trim dataset with
    parameters = yaml.safe_load(open(sys.argv[1],'r'))
    
    ### Read in data to be trimmed and fit
    data = pd.read_csv(parameters['Autocorrelation data'])

    ### Pull out time vector and intensity matrix from the trimmed dataset for fitting
    time = data['Time (µs)'][4:]/1e6
    intensity = data[f"{parameters['Fit well']}" + ' '][4:] # Drop first point since anomalous
    intensity = intensity.dropna()

    ### Open pdf to start writing fits
    pdf_fits = matplotlib.backends.backend_pdf.PdfPages('autocorrelation_fits_MonteCarloErrors' + '.pdf')

    ### Make model and initial parameters
    gmodel = Model(autocorrelation)
    fit_params = gmodel.make_params(D=1e-10, B=1.0, beta=0.2, mu2=1e7)

    ### Set up plot
    fig_handle, axis_handle = plt.subplots(1,1)
    axis_handle.set_xlabel('log(time s)')
    axis_handle.set_ylabel('Intensity')
    axis_handle.set_title(parameters['Fit well'])
    
    # Optimal fit to experimental data, set up MC dictionaries
    opt_fit = gmodel.fit(intensity,fit_params,t=np.array(time))
    MC_dict = {k:[] for k in opt_fit.params.keys()}
    errors = {k+' error':[] for k in MC_dict.keys()}
    RMSD = np.sqrt(opt_fit.chisqr/len(intensity))
    perfect_data = opt_fit.best_fit
    
    counter = 1
    for x in range(parameters['MC iterations']):
        
        print(f"######### The current Monte-Carlo iteration is: {counter} #########")
        
        perturbed_data = perfect_data
        perturbed_data = perturbed_data + np.random.normal(scale=RMSD, size=np.size(perturbed_data))
        
        perturbed_result = gmodel.fit(perturbed_data,opt_fit.params,t=np.array(time))
        
        {MC_dict[k].append(perturbed_result.params[k]) for k in perturbed_result.params.keys()}
        
        # Plot each MC fit
        axis_handle.semilogx(time,opt_fit.best_fit,'r')
        
        counter += 1
    
    # Plot data and optimal fit
    axis_handle.semilogx(time,intensity,'ko',markersize=0.5)
    axis_handle.semilogx(time,opt_fit.best_fit,'b--')
    pdf_fits.savefig(fig_handle)
    
    for k in MC_dict.keys():
        errors[k+' error'].append(1.96*np.std(MC_dict[k]))
        
        # Plot MC histograms
        MC_fig, MC_ax = plt.subplots(1,1)
        MC_ax.set_ylabel('Count')
        MC_ax.set_xlabel('Value')
        MC_ax.set_title(f"Parameter: {k}")
        MC_ax.hist(MC_dict[k],bins=50)
        pdf_fits.savefig(MC_fig)
        plt.close(MC_fig)
    
    pdf_fits.close()

    for k1,k2 in zip(opt_fit.params.keys(),errors.keys()):
        print(f"{k1} = {opt_fit.params[k1].value} +/- {errors[k2][0]}, 95% confidence interval e.g. +/-1.96*std")
        
    with open('MC_parameter_dictionary_autocorrelationdata.pickle','wb') as f:
        pickle.dump(MC_dict,f,protocol=pickle.HIGHEST_PROTOCOL)
        
    print('\n ##### You dropped the first 4 anomalous data points from intensities! #####')
    

### Calculate scattering vector using detector angle and wavelength
def scattering_vector():
    
    ### Define constants and instrument parameters for Wyatt DynaPRO DLS plate reader
    n = 1.3347;
    wavelength = 824e-9;
    theta = (150)*(pi/180);
    
    q = (4*pi*n/wavelength)*np.sin(theta/2);
    
    return q

### Generate autocorrelation function by fitting a diffusion coefficient
def autocorrelation(t, D, B, beta, mu2):
    
    q = scattering_vector()
    g2 = B + beta*np.exp(-2.*D*q**2*t)*((1 + (mu2/2.)*t**2)**2);
    
    return g2

main()
