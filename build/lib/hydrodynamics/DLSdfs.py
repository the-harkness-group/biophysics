#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:46:11 2019

@author: robertharkness
"""

import sys
import numpy as np
import pandas as pd
import yaml

### Read DLS fit params file and yaml file containing list of experimental parameters
data = pd.read_csv(sys.argv[1])
with open(sys.argv[2],'r') as stream:
    params = yaml.safe_load(stream)
    
for Sample in params['Samples']:
    holder_df = pd.DataFrame()
    
    for well in params['Samples'][Sample]['Wells']:
        sub_df = data[data['Wells'] == well]
        holder_df = pd.concat([holder_df,sub_df])
    
    holder_T_df = pd.DataFrame()
    for Temperature in Temperatures:
        sub_T_df = holder_df[round(holder_df['Temperature']) == Temperature]
        sub_T_df = pd.concat([holder_T_df,sub_T_df])

    
