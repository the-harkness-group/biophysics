#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:55:13 2019

@author: robertharkness
"""

import sys
import pandas as pd

data = pd.read_csv(sys.argv[1])
omit_peaks = pd.read_csv(sys.argv[2])

clean_data = data[~data.assignment.isin(omit_peaks['Peak name'])]
#clean_data = data[data.ASS.str.contains('LV') | data.ASS.str.contains('I') | data.ASS.str.contains('M')]

clean_data.to_csv('clean_peaks.csv')