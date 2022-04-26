#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 21:35:29 2019

@author: robertharkness
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import yaml
#import glob
import matplotlib.backends.backend_pdf

#with open(sys.argv[1],'r') as stream:
#    params = yaml.safe_load(stream)
#    filenames = params['filenames']

### Fits from NMR pipe
#filenames = glob.glob('*.list')
#file1 = pd.read_fwf(filenames[0],sep=" ")
#peak_names = file1['Assignment'][1:]

#peak_df = pd.DataFrame()
#for fname in filenames:
#    df = pd.read_fwf(fname,sep=" ")
#    df = df[1:]
#    peak_df = pd.concat([peak_df,df],axis=0)


### Fits from peakipy
data = pd.read_csv(sys.argv[1])
groups = data.groupby('assignment')

hTRF1 = np.array([0,25,47,86,134,246,357,445,466,493])

pdf = matplotlib.backends.backend_pdf.PdfPages(sys.argv[2])
for ind, group in groups:
    print(group)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(hTRF1,group.amp,yerr=group.amp_err,fmt="o",label=group.assignment.iloc[0])
    ax.set_xlabel('[hTRF1] ' '$\mu$M')
    ax.set_ylabel('Peak volume')
    ax.legend()
    
    pdf.savefig(fig)

pdf.close()