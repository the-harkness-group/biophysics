import pandas as pd

class Experiment:
    """Represents a physical experiment and is passed to the optimizer and error minimization
    classes for running fits of models and subsequent analysis of the fitted parameter errors.
    Data set to be read must be .csv."""

    def __init__(self, technique, data_file, x, y, **kwargs):

        self.data_file = data_file
        self.technique = technique
        self.x = x
        self.y = y
        self.__dict__.update(kwargs)

    def load_data(self):

        self.data = pd.read_csv(self.data_file)
