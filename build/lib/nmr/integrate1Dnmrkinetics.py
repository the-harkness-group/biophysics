#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:14:30 2020

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


### SCRIPT FOR PROCESSING AND FITTING ISOTHERMAL FOLDING KINETICS DATA USING
### A LASER SETUP AND PHOTOLABILE BLOCKING GROUPS FOR G-QUADRUPLEXES
### BY NMR
def main():
    
    # Read in filenames to plot, prepare dictionary for nmrglue processing
    params = yaml.safe_load(open(sys.argv[1],'r'))
    files = params['files']
    
    # Set up experimental parameters and kinetic parameters
    window = params['add_window']
    time_inc = params['time_increment']
    time_shift = int(params['laser_after']/window)
    total_points = params['data_points']
    ppm_limits = params['ppm_limits']
    sum_lower = params['lower_sum_range']
    sum_upper = params['upper_sum_range']
    laser_after = params['laser_after']
    
    # Set up plot params
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    colors = ['#1f78b4','#b2df8a']
    lw = 1.5
    
    # Make plot figure and axes    
    kin_fig = plt.figure(figsize=(11,8))
    kin_ax = kin_fig.add_subplot(111)
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    
    # Set up kinetic times
    points = np.linspace(1,total_points/window,total_points/window)
    times = time_inc*window*points
    print(times)
    
    # Set up fitting
    gmodel = Model(exponential)
    
    # Iterate through 1D data files, generate Bruker dictionaries according to spectral parameters
    for n,f in enumerate(files):
        print('File: ',n,f)
        dic, data = ng.pipe.read(f)
        uc0 = ng.pipe.make_uc(dic,data,dim=0)
        uc1 = ng.pipe.make_uc(dic,data,dim=1)

        #slidingwindow_data = []
        #for x in range(data.shape[0]-window):
        #slidingwindow_data.append(np.sum(data[x:x+window,:],0))
        #print(slidingwindow_data[0])
        
        # Sum intensities from each FID according to the chosen window size, total time points (e.g. 512 FIDs) has to have zero modulus i.e. divisible by 2
        # Done to improve S/N
        sum_data = []
        for x in range(int(data.shape[0]/window)):
            sum_data.append(np.sum(data[x*window:(x*window)+window,:],0))
        
        # Sum intensities over specified ppm range for kinetic trajectories, ~10-12 ppm for GQs
        kin_data = []
        for x in range(int(data.shape[0]/window)):
            #print(np.sum(sum_data[x][uc1("12.5ppm"):uc1("10.0ppm")]))
            kin_data.append(np.sum(sum_data[x][uc1(sum_upper):uc1(sum_lower)]))
            ax[n].plot(uc1.ppm_scale(),sum_data[x])
        
        result = gmodel.fit(kin_data[time_shift-1:], xx=times[time_shift-1:]-times[time_shift-1], a=kin_data[time_shift-1], b=kin_data[-1]-kin_data[time_shift-1], k=0.1)
        print(result.fit_report())
        
        kin_ax.plot(times-time_inc*laser_after,kin_data,'o',label=f"{f} data",color=colors[n])
        kin_ax.plot(times[time_shift-1:]-time_inc*laser_after,result.best_fit,'--',label=f"{f} fit",color=colors[n])
        ax[n].set_xlim(ppm_limits)
        ax[n].set_ylim(0,2e5)
        ax[n].set_xlabel('$^{1}H$ ppm')
        ax[n].set_ylabel('Intensity')
        ax[n].set_title(f"{f} NMR data",fontweight='bold')
        kin_ax.set_xlabel('time s')
        kin_ax.set_ylabel('Summed intensity')
        kin_ax.legend()
    #        ax1.text(0.00, 0.95,'c',horizontalalignment='left', verticalalignment='top',transform=ax1.transAxes,fontsize=30,fontweight='bold')
            
    #Set axis properties for each plot
    for idx, ax in enumerate(ax):
        ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True)
        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)
            item.set_fontweight('bold')
            
    kin_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True)
    for item in ([kin_ax.xaxis.label, kin_ax.yaxis.label] + kin_ax.get_xticklabels() + kin_ax.get_yticklabels()):
            item.set_fontsize(14)
            item.set_fontweight('bold')
    
    plt.tight_layout()
    plt.show()

def exponential(xx, a, b, k):
    
    return a + b*(1 - np.exp(-k*xx))
    
main()