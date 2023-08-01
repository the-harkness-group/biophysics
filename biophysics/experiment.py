import pandas as pd
from copy import deepcopy

class Experiment:
    """ Represents a physical experiment and is passed to the optimizer and error minimization
    classes for running fits of models and to the error analyzer class for subsequent analysis 
    of the fitted parameter errors. Implements the data in the form of a pandas dataframe.
    Data set to be read must be .csv. """

    def __init__(self, experiment_type=None, data_file=None, construct=None, x=None, y=None, **kwargs):
        self.experiment_type = experiment_type
        self.data_file = data_file
        self.construct = construct
        self.x = x
        self.y = y
        for k in kwargs:
            setattr(self, k, kwargs[k])

        try:
            self.load_data()
        except:
            print('Cannot load data file; data_file variable set to something other than .csv file')
            pass

    def load_data(self):
        self.data = pd.read_csv(self.data_file)

    def filter_data(self, expression):
        self.data = self.data.query(expression)

    def sparse_data(self, sparsing=1):
        self.data = self.data.iloc[::sparsing][:]

    def scale_concentrations(self, scaling_factor=1e-6):
        variables= ['Protein', 'Receptor', 'Substrate', 'Ligand']
        for v in variables:
            if v in self.data.columns:
                self.data.loc[:,v] = self.data.loc[:,v]*scaling_factor