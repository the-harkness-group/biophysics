#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 17:10:25 2020

@author: robertharkness
"""


import sys
import pandas as pd
import yaml
import numpy as np
import nmrglue as ng
import matplotlib
import matplotlib.pyplot as plt


### SCRIPT FOR INTEGRATING 1D NMR SPECTRA INTENSITIES IN DEFINED RANGES
### CAN CHOOSE REFERENCE SIGNAL SUCH AS DSS TO BE INTEGRATED FOR COMPARISON OF NMR SPECTRA IN DIFFERENT CONDITIONS
def main():
    
    # Read in filenames to plot, prepare dictionary for nmrglue processing
    params = yaml.safe_load(open(sys.argv[1],'r'))
    files = params['files']
    
    # Set up experimental parameters and kinetic parameters
    ppm_limits = params['ppm_limits']
    sum_lower = params['lower_sum_range']
    sum_upper = params['upper_sum_range']
    reference_flag = params['reference_flag'] # If you have a reference signal you also want to integrate and compare intensities with in different 1Ds
    
    # Set up plot params
    # Make plot figure and axes    
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    lw = 2
    fig, axs = plt.subplots(1,len(files)+1,figsize=(15,5)) # Number of axes is number of files + 1 extra for histogram comparison of intensities
    
    # Iterate through 1D data files, generate dictionaries according to spectral parameters
    for n,f in enumerate(files):
        
        print('File: ',n,f)
        dic, data = ng.pipe.read(f)
        uc0 = ng.pipe.make_uc(dic,data,dim=0)
        
        axs[n].plot(uc0.ppm_scale(),data,linewidth=lw) # Plot 1D NMR data on axis n
        
        # Sum intensities over specified ppm range
        sum_data = np.sum(data[uc0(sum_upper):uc0(sum_lower)])
        axs[-1].bar(2*n,sum_data) # Plot as bar on last axis in figure
            
        if reference_flag == 'y':
            
            ref_ppm_lower = params['reference_ppm_lower']
            ref_ppm_upper = params['reference_ppm_upper']
            
            ref_sum_data = np.sum(data[uc0(ref_ppm_upper):uc0(ref_ppm_lower)])
            
            axs[-1].bar(2*n+1,ref_sum_data)
            
        axs[n].set_xlim(ppm_limits)
        axs[n].set_xlabel('$^{1}H$ ppm')
        axs[n].set_ylabel('Intensity')
        axs[n].set_title(f"{f} NMR data",fontweight='bold')
        axs[n].yaxis.get_offset_text().set_fontsize(12)
        axs[n].yaxis.get_offset_text().set_fontweight('bold')
        axs[-1].yaxis.get_offset_text().set_fontsize(12)
        axs[-1].yaxis.get_offset_text().set_fontweight('bold')
        axs[-1].get_xticklabels()
        axs[-1].set_xticklabels(params['barplot_xlabels'])
    
    # Set axis properties for each plot
    for idx, ax in enumerate(axs):
        ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)
            item.set_fontweight('bold')
            
    plt.tight_layout()
    plt.show()


main()