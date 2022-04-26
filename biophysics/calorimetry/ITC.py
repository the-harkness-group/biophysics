#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: robertharkness
"""

import copy
import pandas as pd

class ITC(): # Make ITC class object for convenient manipulation of ITC experiment parameters and data

    def __init__(self, experiment_file=None, blank_file=None): # Initialize with data filenames, read data

        self.Experiment_file = experiment_file # Experiment name
        self.Blank_file = blank_file # Blank experiment to subtract dilution heats

        if experiment_file is None:
            print('No experiment specified, cannot read experiment data...')
        if blank_file is None:
            print('No blank file specified, cannot read blank data...')
        if experiment_file is not None: # Read experiment data
            Data_dict = {'Injection number':[],'Time s':[],'Power ucal/s':[],'Temperature C':[]}
            Injection_dict = {'Number':[],'Volume uL':[],'Duration s':[],'Spacing s':[]}
            symbols = ['#','$','?','@','%'] # These markers denote lines for experimental and instrumental 
            # parameters and settings that occur prior to the actual experiment data in the .itc file
            counter = -1 # Initial counter for lines prior to experiment data
            with open(self.Experiment_file,'r') as f1:
                    for index,line in enumerate(f1):
                        if index == 1: # Total number of injections
                            number_injections = int(line.split(' ')[1])
                        if index == 3: # Set temperature
                            self.Temperature = float(line.split(' ')[1])
                        if index == 4: # Length of pre-equilibration delay
                            self.Equilibration = float(line.split(' ')[1])
                        if index == 5: # Stirring speed in rpm
                            self.Stirring = float(line.split(' ')[1])
                        if index == 6: # Power offset
                            self.dP = float(line.split(' ')[1])
                        if (index >= 10) and (index <= 10 + int(number_injections) - 1): # Individual injection parameters
                            Injection_dict['Number'].append(int(index - number_injections + 1))
                            Injection_dict['Volume uL'].append(float(line.split(' ')[1]))
                            Injection_dict['Duration s'].append(float(line.split(' ' )[3]))
                            Injection_dict['Spacing s'].append(float(line.split(' ')[5]))
                        if index == 21:
                            self.Syringe = float(line.split(' ')[1])*1000 # Syringe concentration, convert default mM to uM
                        if index == 22:
                            self.Cell = float(line.split(' ')[1])*1000 # Cell concentration, convert default mM to uM
                        if line[0] == '@': # Signifies start of pre-equilibration for @0, and start of actual injections for @>0
                            counter += 1 # Injection counter, pre-equilibration is 0
                            line_counter = index
                        if (line[0] not in symbols) and (index > line_counter): # Pre-equilibration (lines after @0) and injection data (lines after @>0)
                            Data_dict['Injection number'].append(counter)
                            Data_dict['Time s'].append(float(line.split(',')[0]))
                            Data_dict['Power ucal/s'].append(float(line.split(',')[1]))
                            Data_dict['Temperature C'].append(float(line.split(',')[2]))
            
            self.Experiment = pd.DataFrame(Data_dict)
            self.Injections = pd.DataFrame(Injection_dict)

            # Read blank data, don't need to store experiment parameters as above since these should be identical to the experiment
            if blank_file is not None:
                with open(self.Blank_file,'r') as f2:
                    Blank_dict = {'Injection number':[],'Time s':[],'Power ucal/s':[],'Temperature C':[]}
                    counter = -1 # Initial counter for lines prior to experiment data
                    for index, line in enumerate(f2):
                        if line[0] == '@': # Signifies start of pre-equilibration for @0, and start of actual injections for @>0
                            counter += 1 # Injection counter, pre-equilibration is 0
                            line_counter = index
                        if (line[0] not in symbols) and (index > line_counter): # Pre-equilibration (lines after @0) and injection data (lines after @>0)
                            Blank_dict['Injection number'].append(counter)
                            Blank_dict['Time s'].append(float(line.split(',')[0]))
                            Blank_dict['Power ucal/s'].append(float(line.split(',')[1]))
                            Blank_dict['Temperature C'].append(float(line.split(',')[2]))
                
                self.Blank = pd.DataFrame(Blank_dict)

    def blank_subtract(self): # Do blank subtraction of experiment data
        try:
            self.Subtracted = copy.deepcopy(self.Experiment)
            self.Subtracted['Power ucal/s'] = self.Experiment['Power ucal/s'] - self.Blank['Power ucal/s']
        except:
            print('Blank subtraction failed, either experiment or blank data is not defined...')

    # def baseline_correction(self):

    # def dilution_correction(self):

    # def integrate_peaks(self):
            

