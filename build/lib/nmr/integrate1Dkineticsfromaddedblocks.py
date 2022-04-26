#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 21:43:42 2020

@author: robertharkness
"""

import sys
import pandas as pd
import yaml
import numpy as np
import nmrglue as ng
import matplotlib
import matplotlib.pyplot as plt
import lmfit
from lmfit import Model
import glob


### SCRIPT FOR PROCESSING AND FITTING ISOTHERMAL FOLDING KINETICS DATA USING
### A LASER SETUP AND PHOTOLABILE BLOCKING GROUPS FOR G-QUADRUPLEXES
### BY NMR
def main():
    
    # Read in filenames to plot, prepare dictionary for nmrglue processing
    params = yaml.safe_load(open(sys.argv[1],'r'))
    directories = params['directories']
    
    # Set up plot params
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    colors = ['#1f78b4','#b2df8a']

    # Make plot figure and axes    
    kin_fig = plt.figure(figsize=(7,4))
    kin_ax = kin_fig.add_subplot(111)
    
    # Set up fitting
    gmodel = Model(exponential)
    
    # Iterate through 1D data files, generate Bruker dictionaries according to spectral parameters
    for n,d in enumerate(directories):
        print('Directory: ',n+1,d)
        
        # Set up experimental parameters and kinetic parameters
        numberblocks = params['directories'][d]['number blocks']
        blocksize = params['directories'][d]['block size']
        time_inc = params['directories'][d]['time_increment']
        sum_lower = params['directories'][d]['lower_sum_range']
        sum_upper = params['directories'][d]['upper_sum_range']
        
        files = ['added' + str(x) + '.fid.ft' for x in range(1,numberblocks+1)]

        kin_data = []
        for nn, f in enumerate(files):
            print('File: ',nn+1,f)
            
            dic, data = ng.pipe.read(f"./{d}/{f}")
            uc = ng.pipe.make_uc(dic,data)
        
            # Sum intensities over specified ppm range for kinetic trajectories, ~10-12 ppm for GQs
            # Set up kinetic times
            times = np.array([x*blocksize*time_inc for x in range(0,numberblocks)])
            
            kin_data.append(np.sum(data[uc(sum_upper):uc(sum_lower)]))
        
        #result = gmodel.fit(kin_data, xx=times, a=kin_data[0], b=kin_data[-1]-kin_data[0], k=0.1) # not normalized fit
        result = gmodel.fit(kin_data-kin_data[0], xx=times, b=kin_data[-1]-kin_data[0], k=0.01) # normalized fit
        print(result.fit_report())
        
        #norm_data = (kin_data - result.params['a'])/result.params['b'] # not normalized data and fit
        #norm_bestfit = (result.best_fit - result.params['a'])/result.params['b']
        norm_data = (kin_data - kin_data[0])/result.params['b'] # normalized data and fit
        norm_bestfit = result.best_fit/result.params['b']
        
        #kin_ax.plot(times,kin_data,'o',label=f"{d} data",color=colors[n]) # not normalized
        #kin_ax.plot(times,result.best_fit,'--',label=f"{d} fit",color=colors[n])
        kin_ax.plot(times,norm_data,'o',label=f"{d} data",color=colors[n]) # normalized
        kin_ax.plot(times,norm_bestfit,'--',label=f"{d} fit, kobs. = {result.params['k'].value}",color=colors[n])
        kin_ax.set_xlabel('time s')
        kin_ax.set_ylabel('Normalized signal')
        #kin_ax.set_ylabel('Summed intensity')
        kin_ax.legend()

    #Set axis properties for each plot            
    kin_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True)
    for item in ([kin_ax.xaxis.label, kin_ax.yaxis.label] + kin_ax.get_xticklabels() + kin_ax.get_yticklabels()):
            item.set_fontsize(14)
            item.set_fontweight('bold')
    
    plt.tight_layout()
    plt.show()

def exponential(xx, b, k):
    
    # return a + b*(1 - np.exp(-k*xx))
    return b*(1 - np.exp(-k*xx)) # normalized 
    
main()