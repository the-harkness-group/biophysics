#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:29:19 2020

@author: robertharkness
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Set up DLS simulation parameters
def main():
    
    # Simulation parameters
    t = np.logspace(-7,-2.5,100)
    D = [3.0e-11,5e-11] # Diffusion constants
    B = 1
    beta = 0.2
    mu2 = 0
    
    # Simulate DLS data
    g2_dict = autocorrelation(t, D, B, beta, mu2)
    
    # Plot simulations
    plotting(g2_dict)
    
    
### Generate autocorrelation function by fitting a diffusion coefficient
def autocorrelation(t, D, B, beta, mu2):
    
    ### Define constants and instrument parameters for Wyatt DynaPRO DLS plate reader
    n = 1.3347
    wavelength = 830e-9 # Manual says plate reader is 830 nm
    theta = (150)*(np.pi/180) # Manual says detector angle is 150 degrees
    
    # Scattering vector
    q = (4*np.pi*n/wavelength)*np.sin(theta/2)
    
    # Autocorrelation function
    g2_dict = {'t':t,'D1':np.empty_like(t),'D2':np.empty_like(t)}
    g2_dict['D1'] = B + beta*np.exp(-2.*D[0]*q**2*t)*((1 + (mu2/2.)*t**2)**2)
    g2_dict['D2'] = B + beta*np.exp(-2.*D[1]*q**2*t)*((1 + (mu2/2.)*t**2)**2)
    
    return g2_dict


# Plot simulations
def plotting(g2_dict):
    
    label_params = {'mathtext.default': 'regular' }
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 3
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(1,figsize=(6.325,4.6))
    ax.plot(np.log10(g2_dict['t']),g2_dict['D1'],'ko')
    ax.plot(np.log10(g2_dict['t']),g2_dict['D2'],'o',color='#4eb3d3')
    ax.set_ylim([0.99,1.21])
    #ax.set_yticks([1, 1.1, 1.2])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel('Autocorrelation')
    ax.set_xlabel('log(time [s])')
    #ax.set_xticks([-7,-5,-3])
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    ax.tick_params(direction='in',axis='both',length=5,width=3,grid_alpha=0.3,bottom=True,top=True,left=True,right=True,labelsize=24)
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.yaxis.get_offset_text().set_fontweight('bold')
    ax.grid(linestyle='dashed',linewidth=3,dash_capstyle='round',dashes=(1,3))
    
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(24)
        item.set_fontweight('bold')
        
    plt.tight_layout()
    plt.show()
    
main()