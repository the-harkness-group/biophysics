import pandas as pd
import numpy as np
from itertools import product
import inspect

class Simulator():

    def __init__(self, experiment_type=None, x=None, y=None):
        self.experiment_type=experiment_type
        self.x = x
        self.y = y

    def set_function(self, func_name, func):
        setattr(self, func_name, func)

    def inspect_function_args(self, func_name):
        func_args = inspect.signature(self.__dict__[func_name]).parameters
        setattr(self, func_name + '_args', {k:None for k in func_args})
    
    def set_function_args(self, func_name, **kwargs):
        for k in kwargs.keys():
            if k in self.__dict__[func_name + '_args']:
                self.__dict__[func_name + '_args'][k] = kwargs[k]

    def evaluate_function(self, func_name, result_name):
        setattr(self, result_name, self.__dict__[func_name](**self.__dict__[func_name + '_args']))

    @staticmethod
    def simulate_dataframe(experiment, x_lower_bound, x_upper_bound, points, x_name=None, iterable_names=None):
        x = np.linspace(x_lower_bound, x_upper_bound, points) # Independent variable
        iterables = [] # Parameters to iterate over that can change for each set of x
        for col in experiment.columns:
            if col in iterable_names:
                iterables.append(experiment[col].unique())

        sim_dict = {col:[] for col in experiment.columns if col in iterable_names}
        sim_dict[x_name] = []
        for i in product(*iterables):
            for j, k in zip(i, iterable_names):
                sim_dict[k].extend([j for l in x])
            sim_dict[x_name].extend([l for l in x])

        return pd.DataFrame(sim_dict)