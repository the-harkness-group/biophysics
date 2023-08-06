#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from copy import deepcopy
from lmfit import Parameters, minimize, report_fit, Minimizer
from scipy.interpolate import griddata
from scipy.stats import f
import time as tt
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from biophysics.plotting.plotting import make_pdf
import pickle
import os
import pyDOE as pyd

class ErrorAnalysis:

    def __init__(self, minimizer_result, monte_carlo_iterations=None, rmsd=None, points=None, monte_carlo_datasets=[]):

        self.minimizer_result = deepcopy(minimizer_result)
        self.opt_params = deepcopy(minimizer_result.params)
        self.monte_carlo_iterations = monte_carlo_iterations # For Monte carlo
        self.rmsd = rmsd
        self.points = points
        self.monte_carlo_datasets = monte_carlo_datasets

    @staticmethod
    def parameter_range(opt_param, stderr, num_points=20):
        # Generate parameter vectors for correlation surfaces

        if stderr != stderr:
            print('Error supplied for generating parameter correlation range is NaN! Defaulting to param/1.3 <= param <= param*1.3...')
            opt_param_range = np.linspace(opt_param/1.3, opt_param*1.3, num_points)
        if stderr > opt_param:
            print('Error supplied for generating parameter correlation range is larger than the parameter! Defaulting to param/1.3 <= param <= param*1.3...')
            opt_param_range = np.linspace(opt_param/1.3, opt_param*1.3, num_points)
        else:
            opt_param_range = np.linspace(opt_param - 20*stderr, opt_param + 20*stderr, num_points)
            if opt_param_range[opt_param_range.argmin()] < 0:
                print('The first value in the parameter correlation range is <0, defaulting to param/1.3 <= param <= param*1.3...')
                opt_param_range = np.linspace(opt_param/1.3, opt_param*1.3, num_points)

        return opt_param_range
    
    def correlation_pairs(self, params_to_correlate=None, optimize_other_params=False):

        self.correlation_pairs = {} # Big dictionary of all parameter pair combinations and their associated Parameters objects for passing to fitting routine
        #params_to_correlate = [k for k in self.opt_params.keys() if self.opt_params[k].vary == True]
        opt_params_copy = deepcopy(self.opt_params)

        for i in range(len(params_to_correlate) - 1): # Need to correlate ith parameter with only the parameters ahead of it, don't need to do last parameter because it gets done along the way
            if self.opt_params[params_to_correlate[i]].stderr is None:
                self.opt_params[params_to_correlate[i]].stderr = 0.01*self.opt_params[params_to_correlate[i]].value
            param_1_range = self.parameter_range(self.opt_params[params_to_correlate[i]].value, self.opt_params[params_to_correlate[i]].stderr, self.points)

            for j in range(i + 1, len(params_to_correlate)):
                if self.opt_params[params_to_correlate[j]].stderr is None:
                    self.opt_params[params_to_correlate[j]].stderr = 0.01*self.opt_params[params_to_correlate[j]].value
                param_2_range = self.parameter_range(self.opt_params[params_to_correlate[j]].value, self.opt_params[params_to_correlate[j]].stderr, self.points)
                self.correlation_pairs[f"{params_to_correlate[i]},{params_to_correlate[j]}"] = {f"{params_to_correlate[i]}":[], f"{params_to_correlate[j]}":[], "Parameter sets":[], "RSS":[], 'Fit results':[], 'Result order':[]}

                for k, param_1 in enumerate(param_1_range): # Iterate over values for each parameter pairing, set the pairs in question to constants, allow params not in correlation pair to be varied

                    for l, param_2 in enumerate(param_2_range):

                        opt_params_copy[params_to_correlate[i]].value = param_1
                        opt_params_copy[params_to_correlate[i]].vary = False
                        opt_params_copy[params_to_correlate[j]].value = param_2
                        opt_params_copy[params_to_correlate[j]].vary = False

                        if optimize_other_params == False: # DO NOT vary other parameters, this means that fit qualities will just be calculated for each parameter pair while the other parameters are fixed at the best fit values, i.e. NO FITTING!
                            for p in opt_params_copy.keys():
                                if p not in [params_to_correlate[i], params_to_correlate[j]]:
                                    opt_params_copy[p].vary == False
                        else: # DO vary the other parameters while the params being correlated are fixed, i.e. FITS WILL BE RUN!
                            for p in opt_params_copy.keys():
                                if p not in [params_to_correlate[i], params_to_correlate[j]]:
                                    opt_params_copy[p].vary == True

                        self.correlation_pairs[f"{params_to_correlate[i]},{params_to_correlate[j]}"]["Parameter sets"].append(opt_params_copy) # Parallel fit results are not in the same order as this

                        opt_params_copy = deepcopy(self.opt_params)
    
    def parameter_correlation_fits(self, expt, sim, func):

        maxParallelProcesses = cpu_count()
        print('')
        print('### Running parameter correlation fits using {} CPU cores. ###'.format(maxParallelProcesses))
        start = tt.time()
        for param_pairs in self.correlation_pairs.keys():
            parameter_sets = self.correlation_pairs[param_pairs]['Parameter sets']
            with ProcessPoolExecutor(max_workers = maxParallelProcesses) as parallelExecution:
                future_results = {}
                for x in list(np.arange(len(parameter_sets))):
                    future_result = parallelExecution.submit(self.parallel_fit_task, parameter_sets[x], expt, sim, func)
                    future_results[future_result] = x
                for future in as_completed(future_results):
                    ax = future_results[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print('%r generated an exception: %s' % (ax, exc))
                    else:
                        print(f'Parameter pair {param_pairs} iteration {ax} completed.')
                        self.correlation_pairs[param_pairs]['Result order'].append(ax)
                        # self.correlation_pairs[param_pairs]['Fit results'].append(result.params)
                        # print(result.params)
                        # self.correlation_pairs[param_pairs]['RSS'].append(result.chisqr)
                        # self.correlation_pairs[param_pairs][param_pairs.split(',')[0]].append(result.params[param_pairs.split(',')[0]].value)
                        # self.correlation_pairs[param_pairs][param_pairs.split(',')[1]].append(result.params[param_pairs.split(',')[1]].value)

                        self.correlation_pairs[param_pairs]['Fit results'].append(result[0])
                        self.correlation_pairs[param_pairs]['RSS'].append(result[1])
                        self.correlation_pairs[param_pairs][param_pairs.split(',')[0]].append(result[0][param_pairs.split(',')[0]].value)
                        self.correlation_pairs[param_pairs][param_pairs.split(',')[1]].append(result[0][param_pairs.split(',')[1]].value)

        end = tt.time()
        print(f"### Elapsed parameter correlation fit time: {end-start} s ###")

    def monte_carlo_fits(self, expt, sim, objective):

        maxParallelProcesses = cpu_count()
        print('')
        print('### Running Monte Carlo fits using {} CPU cores. ###'.format(maxParallelProcesses))
        start = tt.time()
        with ProcessPoolExecutor(max_workers = maxParallelProcesses) as parallelExecution:
            future_results = {}
            for x in list(np.arange(1, self.monte_carlo_iterations + 1)):
                future_result = parallelExecution.submit(self.monte_carlo_parallel_fit_task, self.opt_params, self.rmsd, expt, sim, objective)
                future_results[future_result] = x
            for future in as_completed(future_results):
                ax = future_results[future]
                try:
                    result, monte_carlo_experiment = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (ax, exc))
                else:
                    print(f'Monte Carlo iteration {ax} completed.')
                    print(self.monte_carlo_parameters)
                    {self.monte_carlo_parameters[k].append(result.params[k].value) for k in self.monte_carlo_parameters.keys()}
                    self.monte_carlo_datasets.append(monte_carlo_experiment.data)
        end = tt.time()
        print(f"### Elapsed parameter correlation fit time was {end-start} s ###")

        for k in self.monte_carlo_parameters.keys():
            self.monte_carlo_errors[f"{k} error"] = np.std(self.monte_carlo_parameters[k])

        print('')
        print('### Monte Carlo parameter error estimates ###')
        for k1, k2 in zip(self.monte_carlo_parameters.keys(), self.monte_carlo_errors.keys()):
            print(f"{k1} = {self.opt_params[k1].value} +/- {self.monte_carlo_errors[k2]}")

    def monte_carlo_parameter_dictionary(self):
        
        self.monte_carlo_parameters = {k:[] for k in self.opt_params.keys() if self.opt_params[k].vary == True}
        self.monte_carlo_errors = {f"{k} error":None for k in self.opt_params.keys() if self.opt_params[k].vary == True}

    @staticmethod
    def parallel_fit_task(initial_guess_params, expt, sim, func, min_method='leastsq', print_current_params=False): 

        #result = minimize(objective, initial_guess_params, method = min_method, args=(expt, sim, print_current_params))
        result = func(initial_guess_params, expt, sim)

        return initial_guess_params, result
    
    @staticmethod
    def monte_carlo_parallel_fit_task(initial_guess_params, rmsd, perfect_experiment, sim, objective, min_method='leastsq', print_current_params=False):

        perturbed_experiment = deepcopy(perfect_experiment)
        # Add noise to "perfect" simulated best-fit data according to best-fit RMSD
        perturbed_experiment.data.loc[:,perturbed_experiment.y] = perturbed_experiment.data.loc[:,perturbed_experiment.y] + np.random.RandomState().normal(scale=rmsd, size=len(perturbed_experiment.data.loc[:,perturbed_experiment.y]))
        perturbed_minimizer_result = minimize(objective, initial_guess_params, method = min_method, args=(perturbed_experiment, sim, print_current_params))
        return perturbed_minimizer_result, perturbed_experiment

    def parameter_correlation_surfaces(self, sample_name, confidence_level, dof_num, dof_den):

        pdf = make_pdf(f"{sample_name}_parameter_correlation_surfaces.pdf")
        pair_groups = self.correlation_result_df.groupby('Parameter pair')
        for ind, group in pair_groups:

            x = group['Param 1 value'].values
            y = group['Param 2 value'].values
            z = group['RSS'].values

            xgrid = np.reshape(x, (self.points, self.points))
            ygrid = np.reshape(y, (self.points, self.points))
            zgrid = np.reshape(z, (self.points, self.points))

            conf_cont = self._calculate_confidence_contour(confidence_level, dof_num, dof_den, xgrid, ygrid, zgrid)

            z = (group['RSS'].values - self.minimizer_result.chisqr)/(dof_den)
            zgrid = np.reshape(z, (self.points, self.points))

            fig, ax = plt.subplots(1, 1)
            a = ax.contourf(xgrid, ygrid, zgrid, levels=200, cmap='turbo')
            ax.plot(conf_cont['x'], conf_cont['y'], 'w.', markersize=6, mew=0.5, mec='k')
            ax.plot(self.opt_params[ind.split(',')[0]].value, self.opt_params[ind.split(',')[1]].value, 'X', markersize=10, mew=1, mec='k', mfc='w')
            cbar = fig.colorbar(a, format='%.1d')
            cbar.ax.set_title('$\Delta$RSS$_{red.}$', pad=10)
            x_label = ind.split(',')[0]
            y_label = ind.split(',')[1]
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        pdf.close()

    def parameter_confidence_interval_surfaces(self, sample_name):

        pdf = make_pdf(f"{sample_name}_confidence_interval_surfaces.pdf")
        pair_groups = self.correlation_result_df.groupby('Parameter pair')
        for ind, group in pair_groups:

            x = group['Param 1 value'].values
            y = group['Param 2 value'].values
            z = group['RSS'].values

            delta_chisqr = (z/self.minimizer_result.chisqr) - 1
            num_fixed = 2
            best_dof = self.minimizer_result.nfree

            F = f.cdf(delta_chisqr*(best_dof/num_fixed), num_fixed, best_dof)

            xgrid = np.reshape(x, (self.points, self.points))
            ygrid = np.reshape(y, (self.points, self.points))
            Fgrid = np.reshape(F, (self.points, self.points)) * 100

            fig, ax = plt.subplots(1, 1)
            a = ax.contourf(xgrid, ygrid, Fgrid, levels=21, cmap='Blues_r')
            cbar = fig.colorbar(a, format="%2d", ticks=[0,20,40,60,80,100])
            cbar.ax.set_title('Confidence level %', pad=9)
            x_label = ind.split(',')[0]
            y_label = ind.split(',')[1]
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close()

        pdf.close()

    def _calculate_confidence_contour(self, confidence_level:float, dof_num:int, dof_den:int, xgrid:np.ndarray, ygrid:np.ndarray, zgrid:np.ndarray) -> dict:
        """ Determines the the confidence contour for a model fit using the value of the F distribution with a given confidence level.
        
        Arguments:
            confidence_level: the confidence level to determine the F-statistic, given as e.g. 0.95
            xgrid: grid of x parameter values for the error surface
            ygrid: grid of y parameter values for the error surface
            zgrid: grid of the fit quality values (e.g. RSS) defining the error surface contours

        Returns:
            F-value: float
            confidence_contour: dict containing x, y, and z values that define the set of parameters giving curve fits that are statistically indistinguishable from the best fit. x,y pairs can be plotted on the error surface to define the confidence contour.
        """

        F_crit = f.ppf(confidence_level, dof_num, dof_den)
        RSS_crit = self.minimizer_result.chisqr*(F_crit*(dof_num/dof_den) + 1)

        # First get x,y coordinates of all RSS values that are below the critical RSS
        threshold_values = {'x':[],'y':[], 'z':[]}
        for i in range(len(zgrid[:,0])):
            for j in range(len(zgrid[0,:])):
                if zgrid[i,j] <= RSS_crit:
                    threshold_values['x'].append(xgrid[i,j])
                    threshold_values['y'].append(ygrid[i,j])
                    threshold_values['z'].append(zgrid[i,j])
        
        # Now find only the x,y coordinates that define the edges of the confidence contour, since above includes points within the contour as well
        threshold_values_df = pd.DataFrame(threshold_values)
        groups = threshold_values_df.groupby('x')
        conf_cont = {'x':[], 'y':[], 'z':[]}
        for ind, group in groups:
            conf_cont['x'].append(ind)
            conf_cont['x'].append(ind)
            conf_cont['y'].append(group.y.iloc[0])
            conf_cont['y'].append(group.y.iloc[-1])
            conf_cont['z'].append(group.z.iloc[0])
            conf_cont['z'].append(group.z.iloc[-1])

        return conf_cont

    def monte_carlo_distributions(self, sample_name):

        pdf = make_pdf(f"{sample_name}_MonteCarlo_parameter_distributions_{self.monte_carlo_iterations}_iterations.pdf")
        for k in self.monte_carlo_parameters.keys():
            fig, ax = plt.subplots(1,1)
            ax.hist(self.monte_carlo_parameters[k], bins=50)
            avg = np.mean(self.monte_carlo_parameters[k])
            one_sd = self.monte_carlo_errors[f"{k} error"]
            ax.set_title(f"{k}: {avg} $\pm$ {one_sd} (1 std.)")
            ax.set_xlabel(k)
            ax.set_ylabel('Count')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()

    @staticmethod
    def make_grid_data(x, y, z, resolution=100, contour_method='cubic'):

        x_resample = np.linspace(min(x), max(x), resolution)
        y_resample = np.linspace(min(y), max(y), resolution)

        x_grid, y_grid = np.meshgrid(x_resample, y_resample)
        z_grid = griddata((x, y), z, (x_grid, y_grid), contour_method)

        return x_grid, y_grid, z_grid
    
    def save_parameter_correlation_results(self, sample_name):

        result_dict = {'Parameter pair':[],'Param 1 value':[],'Param 2 value':[],'RSS':[],'Result order':[]}
        for param_pairs in self.correlation_pairs.keys():
            result_dict['Parameter pair'].extend([param_pairs for x in self.correlation_pairs[param_pairs]['RSS']])
            result_dict['Param 1 value'].extend(self.correlation_pairs[param_pairs][param_pairs.split(',')[0]])
            result_dict['Param 2 value'].extend(self.correlation_pairs[param_pairs][param_pairs.split(',')[1]])
            result_dict['RSS'].extend(self.correlation_pairs[param_pairs]['RSS'])
            result_dict['Result order'].extend(self.correlation_pairs[param_pairs]['Result order'])

        result_df = pd.DataFrame(result_dict)
        result_df = result_df.sort_values('Result order')
        result_df.to_csv(f"{sample_name}_parameter_correlation_results.csv")
        self.correlation_result_df = result_df

    def save_monte_carlo_results(self, sample_name):

        monte_carlo_dict = {'Parameter':[], 'Value':[], 'Error':[]}
        for k1, k2 in zip(self.monte_carlo_parameters, self.monte_carlo_errors):
            monte_carlo_dict['Parameter'].append(k1)
            monte_carlo_dict['Value'].append(self.opt_params[k1].value)
            monte_carlo_dict['Error'].append(self.monte_carlo_errors[k2])
        monte_carlo_df = pd.DataFrame(monte_carlo_dict)
        monte_carlo_df.to_csv(f"{sample_name}_MonteCarlo_errors_{self.monte_carlo_iterations}_iterations.csv", index=False)

        if os.path.exists('./monte_carlo_datasets'):
            for i, df in enumerate(self.monte_carlo_datasets):
                df.to_csv(f'./monte_carlo_datasets/monte_carlo_dataset_{i}.csv')
        else:
            os.mkdir('./monte_carlo_datasets')
            for i, df in enumerate(self.monte_carlo_datasets):
                df.to_csv(f'./monte_carlo_datasets/monte_carlo_dataset_{i}.csv')

        self.monte_carlo_error_df = monte_carlo_df

class InitialParameterExplorer:
    """ Used to explore the influence of initial parameter values on the final fit quality in an optimization routiine, i.e., verify that the final result is the global and not a local minimum.
        Implements latin hypercube sampling to draw appropriately spaced sets of parameters for use as different initial conditions in minimization routines with the lmfit package.
    
    Attributes:
        params: (object) lmfit Parameters object containing initial values and constraints (e.g. min/max values, vary=True/False, etc.) that will be updated with each round of new initial parameters to explore.
        num_params: (int) number of parameters
        num_samples: (int) number of parameter samples
        param_bounds: (list)
        lhs_criterion: (str) criterion for latin hypercube sampling, default is None so that points are randomly distributed in their intervals, can also be 'center', 'cm', 'maximin', 'centermaximin', or 'correlation'
        objective: (callable) the objective function containing the expression to be minimized
        objective_args: (tuple) additional arguments for the objective function
        minimizer_method: (str) optimization algorithm used by lmfit
    """

    def __init__(self, params:object, num_params:int, num_samples:int, param_bounds:list, objective:callable, objective_args:tuple, lhs_criterion=None, minimizer_method='leastsq'):

        self.params = params
        self.num_params = num_params
        self.num_samples = num_samples
        self.param_bounds = param_bounds
        self.lhs_criterion = lhs_criterion
        self.objective = objective
        self.minimizer_method = minimizer_method
        self.args = objective_args

        self._latin_hypercube_sampling()

    def _latin_hypercube_sampling(self):
        """ Generates sets of parameters (list of lists) for initializing the minimization routine at different values using latin hypercube sampling.
            The parameters within each set (list) are scaled (using a log10 scale) according to parameter boundaries supplied to the class object upon instantiation.
        """

        lhs_params = pyd.lhs(self.num_params, self.num_samples, self.lhs_criterion)

        def scale_lhs_params():
            scaled_lhs_params = []
            for x, draw in enumerate(lhs_params):
                scaled_lhs_params.append([10**(lhs_params[x][i]*(np.log10(self.param_bounds[i][1]) -  np.log10(self.param_bounds[i][0])) + np.log10(self.param_bounds[i][0])) for i in range(self.num_params)])
            return scaled_lhs_params
        
        self.lhs_params = scale_lhs_params()

    def _make_parameter_object(self, param_values:list, **kwargs:dict):
        """ Returns lmfit Parameters object containing the names, values, and constraints of the parameteres used in the minimization routine.

        Arguments:
            param_values: (list) values of the parameters to be added to the Parameters object
            kwargs: (dict) additional arguments for Parameters object such as min (int or float) and max (int or float) values, or whether the parameter should be fixed (True/False).

        Returns:
            params: (object) contains names, values, and constraints for parameters used in the fitting routine.
        """

        params = Parameters()
        for param_name, param_value in zip(self.param_names, param_values):
            params.add(param_name, param_value, **kwargs)

        return params
    
    def _update_parameter_values(self, param_values:list):
        """ Updates lmfit Parameters object with new initial values for passing to the minimizer. Parameter constraints, e.g. min and vary values remain the same.

            Arguments:
                param_values: (list) values of the parameters to be updated in the Parameters object. Values should be in the same order as the keys of the Parameters object dictionary.
        
        """

        for i, k in enumerate(self.params.keys()):
            self.params[k].value = param_values[i]
    
    def optimize_initial_parameters(self):
        """ Iterates over parameter sets obtained from latin hypercube sampling and performs fits. Stores optimal parameters and the fit quality from each iteration,
            saves a csv file of the result parameter values and their associated fit qualities, and finally prints the optimal fit quality and parameter set for the user.
        """

        fit_qualities = []
        opt_params = []
        for i, x in enumerate(self.lhs_params):
            self._update_parameter_values(x)
            try:
                minimizer_result = minimize(self.objective, self.params, method=self.minimizer_method, args=self.args)
                opt_params.append(minimizer_result.params)
                fit_qualities.append(minimizer_result.chisqr)
                print(f'Initial parameter explorer fit {i} completed.')
            except:
                print(f'Parameter explorer fit {i} failed.')
                print('Storing fit quality for this set of parameters as very poor, RSS = 1e32')
                opt_params.append(self.params)
                fit_qualities.append(1e32)

        save_dict = {param_name:[] for param_name in self.params.keys()}
        save_dict['Fit quality'] = []
        for x, y in zip(opt_params, fit_qualities):
            for p in x:
                save_dict[p].append(x[p].value)
            save_dict['Fit quality'].append(y)
        save_df = pd.DataFrame(save_dict)
        save_df.to_csv('Initial_parameter_results.csv')

        self.best_params = opt_params[np.argmin(fit_qualities)]
        self.best_fit = fit_qualities[np.argmin(fit_qualities)]
        
        print(f'The optimal starting parameters for your fit, with the minimum final RSS = {self.best_fit}, are:')
        for k in self.best_params.keys():
            print(f"{k}: {self.best_params[k].value}")
