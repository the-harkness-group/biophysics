#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:04:11 2021

@author: harkness
"""

import numpy as np
import pickle
import pandas as pd
from lmfit import Parameters
from lmfit import Model
from lmfit import minimize

# Simple fits exploiting lmfit, e.g. for fitting an exponential
def simple_fit(model, guess_params, x, y):

    gmodel = Model(model) # Make model
    params = gmodel.make_params(**guess_params) # Make parameter guesses from input dictionary
    kwargs = {gmodel.independent_vars[0]:x} # Make dictionary of independent variable values
    result = gmodel.fit(y, params, **kwargs) # Do fit of y values using parameters and kwargs containing x values
    print(result.fit_report())

    return result

# Minimization function for more complex fitting fitting
def objective(fit_params, fit_data, wrapper_func, wrapper_args, observe):

    resid = []
    for x in range(len(fit_data)):

        observable = np.array(wrapper_func(fit_params, wrapper_args, fit_data.Temperature.iloc[x], 
        fit_data.Concentration.iloc[x]))
        resid.append(np.array(fit_data[observe].iloc[x]) - observable)
        
    return resid


# Minimization function for fitting, scaled
def scaled_objective(fit_params, fit_data, wrapper_func, wrapper_args, observe):

    resid = []
    for x in range(len(fit_data)):

        observable = np.array(wrapper_func(fit_params, wrapper_args, fit_data.Temperature.iloc[x], 
        fit_data.Concentration.iloc[x]))
        resid.append(np.array((fit_data[observe].iloc[x] - observable)
        /fit_data[observe].iloc[x]))
        
    return resid


# Generate errors by Monte Carlo analysis
def MonteCarloErrors(fit_data, opt_params, fit_constants, wrapper_func,
    wrapper_args, observe, MC_iter, RMSD, MC_objective):
    
    perfect_data = fit_data.copy() # Make copy of dataframe

    for x in range(len(perfect_data)): # Overwrite diffusion constants to be perfect simulated values from best fit params

        observable = np.array(wrapper_func(opt_params, wrapper_args, perfect_data.Temperature.iloc[x], 
        perfect_data.Concentration.iloc[x]))
        perfect_data.loc[(perfect_data.Temperature == perfect_data.Temperature.iloc[x]) &
        (perfect_data.Concentration == perfect_data.Concentration.iloc[x]), observe] = observable

    MC_dict = {k:[] for k in opt_params.keys()} # Make dictionary for Monte Carlo parameters
    errors = {k+' error':[] for k in MC_dict.keys()} # Make dictionary for errors
    
    # Serial implementation
    counter = 1
    for x in range(MC_iter):

        print(f"######### The current Monte-Carlo iteration is: {counter} #########")

        perturbed_data = perfect_data.copy() # Copy perfect data groups
        perturbed_data[observe] = perturbed_data[observe] + np.random.normal(scale=RMSD, size=np.size(perturbed_data[observe])) # Perturb perfect data for MC analysis

        perturbed_result = minimize(MC_objective, opt_params, method='nelder', args=(perturbed_data, 
        wrapper_func, wrapper_args, observe))
        {MC_dict[k].append(perturbed_result.params[k]) for k in perturbed_result.params.keys()}

        counter = counter + 1

    for k in MC_dict.keys():
        errors[k+' error'].append(np.std(MC_dict[k]))

    for k1,k2 in zip(opt_params.keys(),errors.keys()):
        print(f"{k1} = {opt_params[k1].value} +/- {errors[k2][0]}")
        
    with open(f"MC_parameter_dictionary_{MC_iter}iterations_serial.pickle",'wb') as f:
        pickle.dump(MC_dict,f,protocol=pickle.HIGHEST_PROTOCOL)
    
    return MC_dict, errors


### CURRENTLY BROKEN !!! DOES NOT DO SEPARATE THINGS UP TO MAX PROCESSOR # IN PARALLEL, DOES MULTIPLE OF THE SAME THING ###

###########################################
# Required for parallel processing.
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from multiprocessing import RawArray, cpu_count
from functools import partial
###########################################
###########################################
# Shared (global variables) to support parallel execution.
# dict containing the object, its size, and other properties.
shared_perturbed_df = {}
shared_opt_params_str = {}

# Generate errors by Monte Carlo analysis, parralel implementation
def montecarlo_parallel(fit_data, opt_params, fit_constants, wrapper_func,
    wrapper_args, observe, MC_iter, RMSD, MC_objective):
    
    perfect_data = fit_data.copy() # Make copy of dataframe

    for x in range(len(perfect_data)): # Overwrite diffusion constants to be perfect simulated values from best fit params

        observable = np.array(wrapper_func(opt_params, wrapper_args, perfect_data.Temperature.iloc[x], 
        perfect_data.Concentration.iloc[x]))
        perfect_data.loc[(perfect_data.Temperature == perfect_data.Temperature.iloc[x]) &
        (perfect_data.Concentration == perfect_data.Concentration.iloc[x]), observe] = observable

    MC_dict = {k:[] for k in opt_params.keys()} # Make dictionary for Monte Carlo parameters
    errors = {k+' error':[] for k in MC_dict.keys()} # Make dictionary for errors

    ###########################################
    # Create shared memory space.
    # Object/String part of DataFrame.
    perfect_data_object_df = perfect_data.select_dtypes(include = ['object'])
    perfect_data_object = perfect_data_object_df.to_json()  # Only convert strings.
    perfect_data_object_data = RawArray('u', len(perfect_data_object))  # Shared array of utf-8.
    perfect_data_objectS = np.frombuffer(perfect_data_object_data, dtype = 'U1').reshape(len(perfect_data_object))
    np.copyto(perfect_data_objectS, list(perfect_data_object))  # Use list() to preprocess string into characters for np.array.
    # Int64 part of DataFrame.
    perfect_data_int64_df = perfect_data.select_dtypes(include = ['int64'])
    perfect_data_int64 = perfect_data_int64_df.to_numpy(dtype = np.int64, copy = False)
    perfect_data_int64_data = RawArray('q', perfect_data_int64.size)
    perfect_data_int64S = np.frombuffer(perfect_data_int64_data, dtype = np.int64).reshape(perfect_data_int64.shape)
    np.copyto(perfect_data_int64S, perfect_data_int64)
    # Float64 part of DataFrame.
    perfect_data_float64_df = perfect_data.select_dtypes(include = ['float64'])
    perfect_data_float64 = perfect_data_float64_df.to_numpy(dtype = np.float64, copy = False)
    perfect_data_float64_data = RawArray('d', perfect_data_float64.size)
    perfect_data_float64S = np.frombuffer(perfect_data_float64_data, dtype = np.float64).reshape(perfect_data_float64.shape)
    np.copyto(perfect_data_float64S, perfect_data_float64)
    # LMFIT Parameters() object.
    opt_params_str = opt_params.dumps()  # Get LMFIT Parameters() as a string.
    opt_params_data = RawArray('u', len(opt_params_str))  # Shared array of utf-8.
    opt_params_strS = np.frombuffer(opt_params_data, dtype = 'U1').reshape(len(opt_params_str))
    np.copyto(opt_params_strS, list(opt_params_str))  # Use list() to preprocess string into characters for np.array.

    #"""
    # Parallelized code.  (Generic implementation.  Can 'submit' different functions to run concurrently if required.)
    maxParallelProcesses = cpu_count()
    print('Computing Monte-Carlo iterations using {} CPU cores.'.format(maxParallelProcesses))
    map_lmFitTask = partial(lmFitTask, RMSD = RMSD, MC_objective = MC_objective,
                    observe = observe, wrapper_func = wrapper_func, wrapper_args = wrapper_args)
    with ProcessPoolExecutor(max_workers = maxParallelProcesses, \
        initializer = initialize_shared_memory, \
        initargs = (perfect_data.columns.tolist(), \
            perfect_data_object_df.columns.tolist(), len(list(perfect_data_object)), perfect_data_object_data, \
            perfect_data_int64_df.columns.tolist(), perfect_data_int64_df.index, perfect_data_int64_df.shape, perfect_data_int64_data, \
            perfect_data_float64_df.columns.tolist(), perfect_data_float64_df.index, perfect_data_float64_df.shape, perfect_data_float64_data, \
            opt_params_data, len(opt_params_str))) \
        as parallelExecution:
        # index [i] is implicit!
        futureResults = {}  # A dictionary!
        for x in list(np.arange(1, MC_iter + 1)):  # 'counter'
            futureResult = parallelExecution.submit(map_lmFitTask, x)
            futureResults[futureResult] = x
        for future in as_completed(futureResults):
            ax = futureResults[future]
            try:
                perturbed_result = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (ax, exc))
            else:
                print('####### Monte Carlo Iteration ', ax, ' completed. #######')
                {MC_dict[k].append(perturbed_result.params[k]) for k in perturbed_result.params.keys()}
                
    for k in MC_dict.keys():
        errors[k+' error'].append(np.std(MC_dict[k]))

    for k1,k2 in zip(opt_params.keys(),errors.keys()):
        print(f"{k1} = {opt_params[k1].value} +/- {errors[k2][0]}")
        
    with open(f"MC_parameter_dictionary_{MC_iter}iterations.pickle",'wb') as f:
        pickle.dump(MC_dict,f,protocol=pickle.HIGHEST_PROTOCOL)

    return MC_dict, errors

# Shared memory initializer.
def initialize_shared_memory(shared_perturbed_df_columns, \
    shared_perturbed_df_object_columns, shared_perturbed_df_object_shape, shared_perturbed_df_object_data, \
    shared_perturbed_df_int64_columns, shared_perturbed_df_int64_index, shared_perturbed_df_int64_shape, shared_perturbed_df_int64_data, \
    shared_perturbed_df_float64_columns, shared_perturbed_df_float64_index, shared_perturbed_df_float64_shape, shared_perturbed_df_float64_data, \
    shared_opt_params_object, shared_opt_params_size):
    #
    shared_perturbed_df['COLUMNS'] = shared_perturbed_df_columns
    #
    shared_perturbed_df['OBJECT_COLUMNS'] = shared_perturbed_df_object_columns
    # 'OBJECT_INDEX' not required since df is tranmitted as json.
    shared_perturbed_df['OBJECT_SHAPE'] = shared_perturbed_df_object_shape
    shared_perturbed_df['OBJECT_DATA'] = shared_perturbed_df_object_data
    #
    shared_perturbed_df['INT64_COLUMNS'] = shared_perturbed_df_int64_columns
    shared_perturbed_df['INT64_INDEX'] = shared_perturbed_df_int64_index
    shared_perturbed_df['INT64_SHAPE'] = shared_perturbed_df_int64_shape
    shared_perturbed_df['INT64_DATA'] = shared_perturbed_df_int64_data
    #
    shared_perturbed_df['FLOAT64_COLUMNS'] = shared_perturbed_df_float64_columns
    shared_perturbed_df['FLOAT64_INDEX'] = shared_perturbed_df_float64_index
    shared_perturbed_df['FLOAT64_SHAPE'] = shared_perturbed_df_float64_shape
    shared_perturbed_df['FLOAT64_DATA'] = shared_perturbed_df_float64_data
    #
    shared_opt_params_str['STRING'] = shared_opt_params_object
    shared_opt_params_str['SIZE'] = shared_opt_params_size
    
# Task for parallel execution.
def lmFitTask(counter, RMSD, MC_objective, observe, wrapper_func, wrapper_args):
    # Access shared memory.
    # Object/String part of DataFrame.
    perfect_data_object_strS = np.frombuffer(shared_perturbed_df['OBJECT_DATA'], dtype = 'U1').reshape(shared_perturbed_df['OBJECT_SHAPE'])
    perfect_data_object_str = ''.join(map(str, perfect_data_object_strS))  # Convert np.array of characters back into string.
    perfect_data_object_df = pd.read_json(perfect_data_object_str)
    # Int64 part of DataFrame.
    perfect_data_int64S = np.frombuffer(shared_perturbed_df['INT64_DATA'], dtype = np.int64).reshape(shared_perturbed_df['INT64_SHAPE'])
    perfect_data_int64_df = pd.DataFrame(data = perfect_data_int64S, columns = shared_perturbed_df['INT64_COLUMNS'], index = shared_perturbed_df['INT64_INDEX'], dtype = np.int64, copy = False)
    # Float64 part of DataFrame.
    perfect_data_float64S = np.frombuffer(shared_perturbed_df['FLOAT64_DATA'], dtype = np.float64).reshape(shared_perturbed_df['FLOAT64_SHAPE'])
    perfect_data_float64_df = pd.DataFrame(data = perfect_data_float64S, columns = shared_perturbed_df['FLOAT64_COLUMNS'], index = shared_perturbed_df['FLOAT64_INDEX'], dtype = np.float64, copy = False)
    # Concatenate the DataFrame parts into one DataFrame and correct column order.
    perfect_data_df = pd.concat([perfect_data_object_df, perfect_data_int64_df, perfect_data_float64_df], axis = 1, copy = False)
    perfect_data_df = perfect_data_df.reindex(shared_perturbed_df['COLUMNS'], axis = 1, copy = False)
    # opt_params - convert raw array to string; then use loads...
    opt_params_strS = np.frombuffer(shared_opt_params_str['STRING'], dtype = 'U1').reshape(shared_opt_params_str['SIZE'])
    opt_params_str = ''.join(map(str, opt_params_strS))  # Convert np.array of characters back into string.
    opt_params = Parameters() # initialize.
    opt_params.loads(opt_params_str)
    # Compute...
    perturbed_data = perfect_data_df.copy() # Copy perfect data
    perturbed_data[observe] = perturbed_data[observe] + np.random.normal(scale=RMSD, size=np.size(perturbed_data[observe])) # Perturb perfect data for MC analysis
    perturbed_result = minimize(MC_objective, opt_params, method='nelder', args=(perturbed_data, 
        wrapper_func, wrapper_args, observe))
    return perturbed_result

#####main()
#if __name__ == "__main__":  # _ALWAYS_ needed.  See https://stackoverflow.com/questions/419163/what-does-if-name-main-do
#    import sys
#    print('this ran')
#    main()
