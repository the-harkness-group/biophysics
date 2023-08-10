#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:04:11 2021

@author: harkness
"""

import time
import numpy as np
import pickle
import pandas as pd
from lmfit import Parameters, minimize, Model, report_fit
import copy

# Simple fits exploiting lmfit, e.g. for fitting an exponential
def simple_fit(model, guess_params, x, y):

    gmodel = Model(model) # Make model
    params = gmodel.make_params(**guess_params) # Make parameter guesses from input dictionary
    kwargs = {gmodel.independent_vars[0]:x} # Make dictionary of independent variable values
    result = gmodel.fit(y, params, **kwargs) # Do fit of y values using parameters and kwargs containing x values
    print(result.fit_report())

    return result

def fit(fit_data, fit_params, fit_constants, wrapper_func, wrapper_args, observe, 
        objective_func, montecarloerrors, error_iter, method='nelder', run_fit=True, get_errors=True, print_fit=False):

    if run_fit == True: # Fit data
        start = time.time() # Calculate fit run time
        result = minimize(objective_func, fit_params, method, args=(fit_data, wrapper_func, wrapper_args, observe, print_fit))
        finish = time.time()
        print(f"\nThe elapsed fit time is {finish-start}s \n")
        
        # Print optimized parameters, write dataframe
        report_fit(result)
        print('The reduced RSS is:',(result.chisqr/(result.ndata-len(result.params.keys()))))
        fit_params = copy.deepcopy(result.params) # Overwrite initial parameter estimates with optimal fitted values
        save_dict = {'Parameter':[],'Value':[]}
        for k,v in fit_params.items():
            save_dict['Parameter'].append(k)
            save_dict['Value'].append(v.value)
        save_df = pd.DataFrame(save_dict)
        save_df.to_csv(f"optimal_fit_params_{fit_data.Sample.iloc[0]}.csv")

        if get_errors == True:
            start = time.time()
            # Calculate RMSD using fitting.objective since you need residuals! NOT SCALED RESIDUALS
            residuals = np.array(objective(fit_params, fit_data, wrapper_func, wrapper_args, observe)) # For RMSD
            RSS = np.sum(np.square(residuals))
            RMSD = np.sqrt(RSS/result.ndata)
            MC_dict, MC_errors = montecarloerrors(fit_data, fit_params, fit_constants, wrapper_func,
            wrapper_args, observe, error_iter, RMSD, objective_func, method)
            finish = time.time()
            print(f"\nThe total elapsed Monte-Carlo error iteration time is {finish-start}s\n")

        if get_errors == False:
            MC_dict = {}
            MC_errors = {}

    if run_fit == False: # Simulate data, if optimal parameters already known or just want to see curves
        if get_errors == True:
            # Calculate RMSD using fitting.objective since you need residuals! NOT SCALED RESIDUALS
            residuals = np.array(objective(fit_params, fit_data, wrapper_func, wrapper_args, observe)) # For RMSD
            RSS = np.sum(np.square(residuals))
            RMSD = np.sqrt(RSS/len(fit_data))

            start = time.time()
            MC_dict, MC_errors = montecarloerrors(fit_data, fit_params, fit_constants, wrapper_func,
            wrapper_args, observe, error_iter, RMSD=RMSD, MC_objective=objective_func)
            finish = time.time()
            print(f"\nThe total elapsed Monte-Carlo error iteration time is {finish-start}s\n")
 
        if get_errors == False:
            MC_dict = {}
            MC_errors = {}

    return fit_params, MC_dict, MC_errors

# Minimization function for more complex fitting
def objective(params, expt, sim, print_current_params=False):

    sim.set_function_args('model', params=params)
    sim.evaluate_function('model', 'model_result')
    sim.set_function_args('observable', model_df=sim.model_result, fit_params=params)
    sim.evaluate_function('observable', 'simulated_data')
    resid = expt.data[expt.y].values - sim.simulated_data[sim.y].values
    
    if print_current_params is True:
        print("RSS:",np.sum(np.square(resid)))
        params.pretty_print(fmt='e',colwidth=12,columns=['value'])
    
    return resid

# Minimization function for fitting, scale residual,i by data,i
def scaled_objective(fit_params, fit_data, wrapper_func, wrapper_args, observe):

    resid = []
    for x in range(len(fit_data)):

        observable = np.array(wrapper_func(fit_params, wrapper_args, fit_data.Temperature.iloc[x], 
        fit_data.Concentration.iloc[x]))
        resid.append(np.array((fit_data[observe].iloc[x] - observable)
        /fit_data[observe].iloc[x]))
        
    return resid

def rss(params, expt, sim):

    sim.set_function_args('model', params=params)
    sim.evaluate_function('model', 'model_result')
    sim.set_function_args('observable', model_df=sim.model_result, fit_params=params)
    sim.evaluate_function('observable', 'simulated_data')
    rss = np.sum(np.square(expt.data[expt.y].values - sim.simulated_data[sim.y].values))

    return rss