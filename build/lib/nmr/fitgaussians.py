#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:46:48 2020

@author: robertharkness
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
import pandas as pd

# Setup function
def main():
    
    data = pd.read_csv('data.txt', delimiter='\t') # Read data
    data.dropna() # Get rid of entries where no data was measured for those residues, ie proline

    x = data.dropna().iloc[:,0].values # Residue number
    y = data.dropna().iloc[:,1].values # y-value

    fit_params = Parameters()
    fit_params.add('h1',value=6) # Gaussian height
    fit_params.add('c1',value=630) # Gaussian center
    fit_params.add('w1',value=10) # Gaussian width
    fit_params.add('h2',value=6)
    fit_params.add('c2',value=638)
    fit_params.add('w2',value=10)
    fit_params.add('h3',value=6)
    fit_params.add('c3',value=660)
    fit_params.add('w3',value=10)
    fit_params.add('lambda0',value=1,min=1) # Baseline persistence length, minimum should be 1 residue long
    fit_params.add('Rint',value=1) # Baseline offset
    
    result = minimize(objective, fit_params, method='nelder', args=(x,y)) # Run fit
    report_fit(result)
    R_best, baseline_best = four_gaussians(result.params, x)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(x,y,'ko')
    ax.plot(x,R_best,'r')
    ax.plot(x,baseline_best,'b')
    ax.set_xlabel('Residue number')
    ax.set_ylabel('NOE Intensity')
    plt.savefig('fit.png',format='png') # Save figure
    plt.show()
    
    final_dict = {}
    for k in result.params.keys():
        final_dict[k] = result.params[k].value
    fit_df = pd.DataFrame(final_dict,index=[0])
    fit_df.to_csv('fitted_parameters.csv') # Save optimized fit parameters
    
        
# Fit objective for calculating fit quality
def objective(fit_params, x, y):
    
    y_sim, baseline = four_gaussians(fit_params, x)
    residuals = y_sim - y
    
    #fig = plt.figure(figsize=(11,7))
    #ax = fig.add_subplot(111)
    #ax.plot(x,y,'ko')
    #ax.plot(x,y_sim,'r')
    #ax.plot(x,baseline,'bo')
    
    #plt.ion()
    #plt.pause(0.00001)
    #plt.ioff()
    #plt.close()
    
    return residuals


# Calculate gaussian
def gaussian(x, height, center, width):
    return height*np.exp(-np.abs(x - center)**2/width)


# Calculate Ri
def four_gaussians(fit_params, x):
    
    h1 = fit_params['h1'].value # Unpack fit parameters
    c1 = fit_params['c1'].value
    w1 = fit_params['w1'].value
    h2 = fit_params['h2'].value
    c2 = fit_params['c2'].value
    w2 = fit_params['w2'].value
    h3 = fit_params['h3'].value
    c3 = fit_params['c3'].value
    w3 = fit_params['w3'].value
    lambda0 = round(fit_params['lambda0'].value) # Force persistence length to be an integer
    Rint = fit_params['Rint'].value
    
    baseline = np.zeros(len(x))
    R = np.zeros(len(x))
    
    x_new = np.arange(len(x)) + x[0] # Make new list of residue numbers that removes gaps to fix the baseline calculation

    for index1, value1 in enumerate(x_new):
        
        baseline[index1] = np.sum([np.exp(-np.abs(value1-value2)/lambda0) for value2 in x_new])
        baseline[index1] = baseline[index1]*Rint
        R[index1] = baseline[index1] + gaussian(value1, h1, c1, w1) + gaussian(value1, h2, c2, w2) + gaussian(value1, h3, c3, w3)
        
    return R, baseline


main() # Call main function