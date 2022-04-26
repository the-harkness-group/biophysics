#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:46:45 2019

@author: robertharkness
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

# Correlate NMR peak intensities for two experiments, i.e. HMQCs before and after
# a longer experiment such as CPMG relaxation-dispersion measurements

# NMR Pipe
#data1 = pd.read_csv(sys.argv[1],header=None,skiprows=6,delim_whitespace=True)
#data1.columns = ['peak_number','x','y','dx','dy','x_ppm','y_ppm','x_hz','y_hz','xw','yw','xw_hz','yw_hz','x1','x3','y1','y3','height','dheight','vol','pchi2','type','ass','clustid','memcnt']
#data2 = pd.read_csv(sys.argv[2],header=None,skiprows=6,delim_whitespace=True)
#data2.columns = ['peak_number','x','y','dx','dy','x_ppm','y_ppm','x_hz','y_hz','xw','yw','xw_hz','yw_hz','x1','x3','y1','y3','height','dheight','vol','pchi2','type','ass','clustid','memcnt']

# Peakipy
data1 = pd.read_csv(sys.argv[1])
data2 = pd.read_csv(sys.argv[2])

bestline_params = np.polyfit(data1.amp,data2.amp,1)
linx = np.linspace(min(data1.amp),max(data2.amp),100)
bestline = bestline_params[0]*linx + bestline_params[1]
yequalsx = np.linspace(min(data1.amp),max(data2.amp))

# Plot settings
pdf = matplotlib.backends.backend_pdf.PdfPages("Pre_post_intensity_correlations.pdf")
label_params = {'mathtext.default': 'regular'}
plt.rcParams.update(label_params)
plt.rcParams['axes.linewidth'] = 2

# Plot correlation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data1.amp,data2.amp,'ko')
ax.plot(linx,bestline,'r--',LineWidth=2,label=f"y = {np.round(bestline_params[0],2)}x + b")
ax.plot(yequalsx,yequalsx,'k--',LineWidth=2,label='y = x')
ax.set_xlabel(f"{sys.argv[1]} Intensity",fontsize=14)
ax.set_ylabel(f"{sys.argv[2]} Intensity",fontsize=14)
ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
ax.yaxis.major.formatter._useMathText = True
ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
ax.set_ylim(1e7,1e9)
ax.set_xlim(1e7,1e9)
ax.legend(loc='upper left',frameon=False)
pdf.savefig(fig)
pdf.close()
