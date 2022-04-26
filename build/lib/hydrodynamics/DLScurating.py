#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:25:58 2019

@author: robertharkness
"""

##########################################################################
### THIS SCRIPT HAS BEEN MADE OBSOLETE AS ITS CONTENTS HAVE BEEN ADDED ###
### TO DLSfit.py SO THAT EVERYTHING HERE HAPPENS AFTER FITTING THE DLS ###
### AUTOCORRELATION FUNCTIONS ############################################
##########################################################################

import sys
import numpy as np
import yaml
import pandas as pd

print('####### This overwrites the fit parameter dataframe by adding sample, concentration, and temperature columns! #######') 

data = pd.read_csv(sys.argv[1])
Sample_dict = yaml.safe_load(open(sys.argv[2],'r'))

newdata = data.copy()
for sample in Sample_dict['Samples']:
    
    print(f"Sample: {sample}",Sample_dict['Samples'][sample])
    for idx_well, well_dict in enumerate(Sample_dict['Samples'][sample]['Wells']):
        for idx_data, well_data in enumerate(newdata['Well']):
            if newdata['Well'][idx_data] == well_dict:
                newdata.loc[newdata.Well==well_dict,'Sample'] = sample
                newdata.loc[newdata.Well==well_dict,'Concentration'] = Sample_dict['Samples'][sample]['Concentration'][idx_well]
                newdata.loc[newdata.Well==well_dict,'Temperature'] = np.round(newdata.loc[newdata.Well==well_dict,'Temperature'],1) # Round temperature to 1-decimal place
    
newdata.to_csv(Sample_dict['Curated_outname'])