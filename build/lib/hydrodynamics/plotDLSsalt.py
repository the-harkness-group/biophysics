#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 08:25:33 2021

@author: robertharkness
"""

import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import hydrodynamics
import customcolormap
from matplotlib import cm
import numpy as np

def plotDLSsalt():
    
    # Read in params
    params = yaml.safe_load(open(sys.argv[1],'r'))
    data = {}
    sample = params['Plot sample']
    
    # Set up plotting
    plt.style.use('figure')
    #fig, ax = plt.subplots(1,1,figsize=(7,5.83))
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('Temperature \N{DEGREE SIGN}C')
    #ax.set_ylabel('$D_{z}$($\eta _{0}$) $cm^{2}$ $s^{-1}$')
    ax.set_ylabel('$D_{z}$(0 NaCl) $cm^{2}$ $s^{-1}$')
    ax.set_ylim([1e-7,9.0e-7])
    hex_colors = ['#020101','#281112','#4E2128','#733243','#984364','#BD548A','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF'] # Colors
    hex_colors = list(reversed(hex_colors))
    M3_color = '#fc8d59'
    M6_color = '#78c679'
    M12_color = '#9e0142'
    M18_color = '#8856a7'
    M24_color = '#2b8cbe'
    #cmap = customcolormap.get_continuous_cmap(hex_colors,len(params['Sample'][sample]['File']))
    cm_space = np.linspace(0,1,len(params['Sample'][sample]['File']))
    cmap = [cm.Spectral(x) for x in cm_space]
    #cmap = ['#3288bd','#7fbc41','#de77ae']
    
    for index, well in enumerate(params['Sample'][sample]['Well']):
        data[well] = {}
        initialdata = pd.read_csv(params['Sample'][sample]['File'][index]) # Read dataset
        subdata = initialdata[initialdata.Sample == sample] # Get sub-dataframe with only DegP2 data
        data[well]['Data'] = subdata[subdata['Well'] == params['Sample'][sample]['Well'][index]]
        #data[file]['Data'] = subdata[(subdata.Concentration < 205) & (subdata.Concentration > 185)] # Get sub-dataframe with only 200 uM concentration
        data[well]['Viscosity'] = hydrodynamics.viscosity(data[well]['Data'].Temperature.values+273.15, params['Sample'][sample]['Viscosity coefficients'][index]) # Calculate viscosity according to buffer conditions
        data[well]['D3_0'] = hydrodynamics.stokes_diffusion(data[well]['Data'].Temperature.values+273.15, data[well]['Viscosity'], params['Rh_3']) # Calculate trimer D0
        data[well]['D6_0'] = data[well]['D3_0']*2**(-0.227) # Calculate hexamer D0
        data[well]['D12_0'] = data[well]['D3_0']*4**(-0.333) # Calculate 12-mer D0
        data[well]['D18_0'] = data[well]['D3_0']*6**(-0.333) # Calculate 18-mer D0
        data[well]['D24_0'] = data[well]['D3_0']*8**(-0.333) # Calculate 24-mer D0
        data[well]['D_corr'] = data[well]['Data'].D.values*(data[well]['Viscosity']/data[params['Sample'][sample]['Well'][0]]['Viscosity']) # Corrected D for viscosity effects to compare
        
        if index == 0:
            ax.plot(data[well]['Data'].Temperature.values,data[well]['D3_0']*1e4,'--',linewidth=2,color=M3_color) # Plot lines for M3, M6, M24 as references in low salt
            ax.plot(data[well]['Data'].Temperature.values,data[well]['D6_0']*1e4,'--',linewidth=2,color=M6_color)
            ax.plot(data[well]['Data'].Temperature.values,data[well]['D12_0']*1e4,'--',linewidth=2,color=M12_color)
            ax.plot(data[well]['Data'].Temperature.values,data[well]['D18_0']*1e4,'--',linewidth=2,color=M18_color)
            ax.plot(data[well]['Data'].Temperature.values,data[well]['D24_0']*1e4,'--',linewidth=2,color=M24_color)
            #ax.plot(data[well]['Data'].Temperature.values,data[well]['Data'].D.values*1e4,'*',markersize=8,color=cmap[index]) # Plot DLS data uncorrected, reference condition
            ax.plot(data[well]['Data'].Temperature.values,data[well]['D_corr']*1e4,'o',color=cmap[index]) # Plot corrected DLS data, reference condition
            
        if index > 0:
            #ax.plot(data[well]['Data'].Temperature.values,data[well]['Data'].D.values*1e4,'*',markersize=8,color=cmap[index]) # Plot DLS data uncorrected, higher salt condition
            ax.plot(data[well]['Data'].Temperature.values,data[well]['D_corr']*1e4,'o',color=cmap[index]) # Plot correctd DLS data for higher salt conditions
    
    fig.tight_layout()
    fig.savefig(f"{params['PDF name']}",format='pdf')

plotDLSsalt()