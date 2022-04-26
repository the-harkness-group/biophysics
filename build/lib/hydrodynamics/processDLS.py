#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:18:50 2019

@author: robertharkness

MAKE SURE THAT THE OUTNAME FOR SAVING IS NOT THE SAME AS THE ORIGINAL FILE OTHERWISE IT WILLOVERWRITE

NEEDS TWO ARGUMENTS AFTER THE SCRIPT CALL: DATA FILE AND YAML FILE CONTAINING WELL INDICES AND NEW FILE NAME FOR SAVING

"""

import pandas as pd
import yaml
import sys

# Read in DLS data
data = pd.read_csv(sys.argv[1])

# Get rid of first point from each column since this is anomalous
data = data[1:]

# Get list of wells from yaml file
parameters = yaml.load(open(sys.argv[2],'r'))
print(parameters['Wells'])
wells = [parameters['Wells'][x] + '_' for x in range(len(parameters['Wells']))]
print(wells)
#wells = parameters['Wells']

# Get list of well names with temperatures from the dataframe
trimmed_wells = []
for well in wells:
    trimmed_wells = trimmed_wells + [col for col in data.columns if well in col]

# Generate trimmed DLS dataset and concatenate with the experimental time from
# the original DLS dataset
trimmed_data = data[trimmed_wells]
trimmed_data = pd.concat([data[data.columns[0]], trimmed_data], axis=1)

# Write trimmed DLS data file to new spreadsheet file
trimmed_data.to_csv(parameters['Outname'])

    

