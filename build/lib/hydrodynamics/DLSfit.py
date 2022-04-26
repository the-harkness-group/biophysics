#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:01:18 2019

@author: robertharkness
"""

import sys
import yaml
import pandas as pd
import numpy as np
from scipy.constants import pi, Boltzmann
from scipy import special
from lmfit import Model
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

############################################################################
##################### FUNCTIONS ############################################
############################################################################

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

### Calculate viscosity of water as a function of temperature using empirically-
### determined equation
def water_viscosity(Temperature):
    
    viscosity = (2.414e-5)*10**(247.8/(Temperature-140))
    
    return viscosity

### Calculate hydrodynamic radius from fitted diffusion coefficient
### Temperature must be in Kelvin
def hydrodynamic_radius(Temperature, D):
    
    viscosity = water_viscosity(Temperature)
    
    Rh = (Boltzmann*Temperature)/(6*pi*viscosity*D)
    
    return Rh

############################################################################
##################### SET UP AND FIT #######################################
############################################################################

### Get list of wells from yaml file to trim dataset with
parameters = yaml.safe_load(open(sys.argv[1],'r'))

print(parameters['Autocorrelation data'])

### Read in data to be trimmed and fit
data = pd.read_csv(parameters['Autocorrelation data'])

### Get rid of first point from each column since this is anomalous
data = data[1:]


wells = []
for k in parameters['Samples'].keys():
    wells = wells + parameters['Samples'][k]['Wells']
wells = [wells[x] + '_' for x in range(len(wells))]

#wells = [parameters['Wells'][x] + '_' for x in range(len(parameters['Wells']))]

### Get list of well names with temperatures from the dataframe
trimmed_wells = []
for well in wells:
    trimmed_wells = trimmed_wells + [col for col in data.columns if well in col]

### Generate trimmed DLS dataset and concatenate with the experimental time from
### the original DLS dataset
trimmed_data = data[trimmed_wells]
trimmed_data = pd.concat([data[data.columns[0]], trimmed_data], axis=1)

### Pull out time vector and intensity matrix from the trimmed dataset for fitting
time = trimmed_data['Time (µs)']/1e6
well_names = trimmed_data.columns[1:]
intensity = trimmed_data[well_names]
intensity = intensity.dropna('columns')

### Set up fit print messages and figure, axis handles if plotting
separator = '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
fail_separator = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

### Open pdf to start writing fits
pdf_fits = matplotlib.backends.backend_pdf.PdfPages('autocorrelation_fits_' + parameters['Date'] + '.pdf')

### Make model and initial parameters
gmodel = Model(autocorrelation)
params = gmodel.make_params(D=1e-10, B=1.0, beta=0.2, mu2=1e7)

### Set up optimized parameter dictionary
opt_params = ['D','D_error','B','B_error','beta','beta_error','mu2','mu2_error']
opt_params_dict = {opt_param:[] for opt_param in opt_params}
opt_params_dict['Well'] = []
opt_params_dict['Temperature'] = []
opt_params_dict['Fit_check'] = []

### Run through data and fit
plt.style.use('figure')
for column in intensity.columns:
    try:
        opt_fit = gmodel.fit(intensity[column],params,t=np.array(time))
        print(f"{separator}\nThe current fit is {column}\n{separator}")
        print(opt_fit.fit_report())
        
        fig_handle = plt.figure()
        axis_handle = fig_handle.add_subplot(111)
        axis_handle.semilogx(time,intensity[column],'ko')
        axis_handle.semilogx(time,opt_fit.best_fit,'r--')
        axis_handle.set_xlabel('log$_{10}$(Time s)')
        axis_handle.set_ylabel('Norm. intensity')
        axis_handle.set_title(column)
        axis_handle.tick_params(axis='x', which='major', pad=12)
        axis_handle.set_xticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
        axis_handle.set_xticklabels(['10$^{-6}$','','','10$^{-3}$','','','10$^{0}$'])
        
        Gamma = opt_fit.params['D']*scattering_vector()**2
        PD = opt_fit.params['mu2']/Gamma**2
        axis_handle.text(0.5,0.95,"D = %.2e $\pm$ %.2e cm$^{2}$ s$^{-1}$\nPD = %.2f"%(opt_fit.params['D']*1e4,opt_fit.params['D'].stderr*1e4,PD),transform=axis_handle.transAxes,va="top")
        
        fig_handle.tight_layout()
        pdf_fits.savefig(fig_handle)
        plt.close()
        
        ### Write fit parameters to dictionary for converting to dataframe
        opt_params_dict['Well'].append(column.split('_')[0])
        Temperature = column.split('_')[1]
        Temperature = Temperature.split('(')[0]
        opt_params_dict['Temperature'].append(Temperature)
        opt_params_dict['Fit_check'].append('Pass')
        
        for k,v in opt_fit.params.items():
            opt_params_dict[k].append(v.value)
            opt_params_dict[k+'_error'].append(v.stderr)
        
    except:        
        fig_handle = plt.figure()
        axis_handle = fig_handle.add_subplot(111)
        axis_handle.semilogx(time,intensity[column],'ko')
        axis_handle.set_xlabel('log$_{10}$(Time s)')
        axis_handle.set_ylabel('Norm. intensity')
        axis_handle.set_title(column + 'Fit failed')
        axis_handle.tick_params(axis='x', which='major', pad=12)
        axis_handle.set_xticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
        axis_handle.set_xticklabels(['10$^{-6}$','','','10$^{-3}$','','','10$^{0}$'])
        
        fig_handle.tight_layout()
        pdf_fits.savefig(fig_handle)
        plt.close()
        
        opt_params_dict['Well'].append(column.split('_')[0])
        Temperature = column.split('_')[1]
        Temperature = Temperature.split('(')[0]
        opt_params_dict['Temperature'].append(Temperature)
        opt_params_dict['Fit_check'].append('Fail')
        
        for k,v in opt_fit.params.items():
            opt_params_dict[k].append(v.value)
            opt_params_dict[k+'_error'].append(v.stderr)
        
        print(f"{fail_separator}\nThe fit failed on column {column}\n{fail_separator}\n")

pdf_fits.close()

###############################################################################
####################### FINAL DATAFRAME PROCESSING ############################
###############################################################################

### Convert fit parameter dictionary to dataframe, calculate additional parameters
### of interest and write to csv file
opt_df = pd.DataFrame(opt_params_dict)
opt_df['Temperature'] = np.round(opt_df['Temperature'].astype(float),1) # Temperature rounded to one decimal place
opt_df['Gamma'] = opt_df['D']*scattering_vector()**2
opt_df['PD'] = opt_df['mu2']/opt_df['Gamma']**2
### Temperature needs to be in Kelvin to calculate Rh
opt_df['Rh'] = hydrodynamic_radius(opt_df['Temperature']+273.15,opt_df['D'])
opt_df['Rh_error'] = opt_df['D_error']*opt_df['Rh']/opt_df['D']

# Append sample names and concentrations to dataframe for writing
print('\n')
print('########### APPENDING CONCENTRATIONS AND SAMPLE NAMES TO FINAL DLS FIT PARAMETER DATAFRAME ###############')
for sample in parameters['Samples']:
    print(f"Sample: {sample}",parameters['Samples'][sample],'\n')
    for idx_well, well_dict in enumerate(parameters['Samples'][sample]['Wells']):
        for idx_data, well_data in enumerate(opt_df['Well']):
            if opt_df['Well'][idx_data] == well_dict:
                opt_df.loc[opt_df.Well==well_dict,'Sample'] = sample
                opt_df.loc[opt_df.Well==well_dict,'Concentration'] = parameters['Samples'][sample]['Concentration'][idx_well]
                #opt_df.loc[opt_df.Well==well_dict,'Temperature'] = np.round(opt_df.loc[opt_df.Well==well_dict,'Temperature'],1) # Round temperature to 1-decimal place, not needed because done above

# Write final dataframe to csv file
opt_df.to_csv(parameters['Outname'])
