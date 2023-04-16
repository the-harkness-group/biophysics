import numpy as np
import pickle
import pandas as pd
from lmfit import Parameters, minimize, Model, report_fit
import copy

# class ErrorAnalysis(Optimization):

#     def __init__(self, ):

#     fit_data: object
#     opt_params: object
#     fit_constants: dict
#     wrapper_func: function
#     wrapper_args: dict
#     observe: str
#     MC_iter: int
#     RMSD: float
#     MC_objective: function
#     method: str

#     def simulate_best_fit(self) -> object:

#         self.best_fit = self.fit_data.copy()

#         for x in range(len(self.best_fit)):

#             observable = np.array(wrapper_func(self.opt_params, wrapper_args, perfect_data.Temperature.iloc[x], 
#             perfect_data.Concentration.iloc[x]))
#             perfect_data.loc[(perfect_data.Temperature == perfect_data.Temperature.iloc[x]) &
#             (perfect_data.Concentration == perfect_data.Concentration.iloc[x]), observe] = observable

def montecarloerrors(fit_data, opt_params, fit_constants, wrapper_func, wrapper_args, observe, MC_iter, RMSD, MC_objective, method='nelder'):
    
    perfect_data = fit_data.copy() # Make copy of dataframe

    for x in range(len(perfect_data)): # Generate best fit data

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

        perturbed_result = minimize(MC_objective, opt_params, method, args=(perturbed_data, 
        wrapper_func, wrapper_args, observe))
        {MC_dict[k].append(perturbed_result.params[k]) for k in perturbed_result.params.keys()}

        counter = counter + 1

    for k in MC_dict.keys():
        errors[k+' error'].append(np.std(MC_dict[k]))

    for k1,k2 in zip(opt_params.keys(),errors.keys()):
        print(f"{k1} = {opt_params[k1].value} +/- {errors[k2][0]}")
        
    with open(f"MC_parameter_dictionary_{MC_iter}iterations_{fit_data.Sample.iloc[0]}_serial.pickle",'wb') as f:
        pickle.dump(MC_dict,f,protocol=pickle.HIGHEST_PROTOCOL)
    
    return MC_dict, errors