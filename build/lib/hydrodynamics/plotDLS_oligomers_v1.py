#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:02:45 2021

@author: robertharkness
"""

import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import customcolormap
import hydrodynamics
import copy

def main():
    
    params = yaml.safe_load(open(sys.argv[1],'r'))
    datasets = {}
    for dataset in params['Apo datasets']:
        datasets[dataset] = pd.read_csv(dataset)
    P2_bound_data = data[data['Well'].isin(params['Wells'])]
    P2_bound_data = copy.deepcopy(P2_bound_data[P2_bound_data['Temperature'] <= 50.5])
    #P2_bound_data = copy.deepcopy(P2_bound_data[P2_bound_data['Concentration'] >= 20])
    P2_bound_groups = copy.deepcopy(P2_bound_data.groupby('Well',sort=False))
    #P2_bound_groups = copy.deepcopy(P2_bound_data.groupby('Concentration'))
    
    plt.style.use('figure')
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('Temperature \N{DEGREE SIGN}C')
    ax.set_ylabel('$\it{D_{z}}$ cm$^{2}$ s$^{-1}$')
    ax.set_ylim([0.8e-7,9e-7])
    ax.set_yticks([2e-7,4e-7,6e-7,8e-7])
    hex_colors = ['#020101','#281112','#4E2128','#733243','#984364','#BD548A','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF']
    hex_colors = list(reversed(hex_colors))
    M3_color = '#fc8d59'
    M6_color = '#78c679'
    M12_color = '#9e0142'
    M18_color = '#8856a7'
    M24_color = '#2b8cbe'
    M30_color = '#dd3497'
    M60_color = 'k'
    cmap = customcolormap.get_continuous_cmap(hex_colors,len(params['Concentrations']))
    
    Temperatures = np.linspace(5,50,19) + 273.15
    
    D_dict = {'D3':np.zeros(len(Temperatures)),'D6A':np.zeros(len(Temperatures)),
              'D12A':np.zeros(len(Temperatures)),'D18A':np.zeros(len(Temperatures)),
              'D24':np.zeros(len(Temperatures)),'D30':np.zeros(len(Temperatures)),
              'D60':np.zeros(len(Temperatures))}
    eta = hydrodynamics.viscosity(Temperatures, params['eta coefficients']) # Get diffusion coeffs as a fxn of temperature
    Do = hydrodynamics.stokes_diffusion(Temperatures, eta, float(params['Trimer Rh']))
    D_dict['D3'] = hydrodynamics.scaled_diffusion(Do, 1, -0.333)        
    D_dict['D6A'] = hydrodynamics.scaled_diffusion(Do, 2, -0.227)
    D_dict['D12A'] = hydrodynamics.scaled_diffusion(Do, 4, -0.333)
    D_dict['D18A'] = hydrodynamics.scaled_diffusion(Do, 6, -0.333)
    D_dict['D24'] = hydrodynamics.scaled_diffusion(Do, 8, -0.333)
    D_dict['D60'] = hydrodynamics.scaled_diffusion(Do, 20, -0.333)
    
    for index, (ind, group) in enumerate(P2_bound_groups):
        
        print(ind,group.Concentration)
        
#        Rh_hTRF1 = 1.965e-9 # From HYDROPRO D0 at 20C in water
#        D0_hTRF1 = hydrodynamics.stokes_diffusion(group.Temperature.values+273.15,
#                eta,Rh_hTRF1)
#        c_hTRF1 = 400e-6 - params['Concentrations'][index]*1e-6
#        M_hTRF1 = 6709.81
#        c_12mer = params['Concentrations'][index]*1e-6/12
#        M_12mer = 46812.6*12
#        Dz_hTRF1 = (D0_hTRF1*c_hTRF1*M_hTRF1**2)/(c_hTRF1*M_hTRF1**2 + c_12mer*M_12mer**2)
#        Dz_corr = group.D.values - Dz_hTRF1
#        ax.plot(group.Temperature,Dz_corr*1e4,'o',color=cmap[index])
        
        ax.plot(group.Temperature,group.D*1e4,'o',color=cmap[index])

    ax.plot(Temperatures-273.15,D_dict['D3']*1e4,'--',color=M3_color)
    ax.plot(Temperatures-273.15,D_dict['D6A']*1e4,'--',color=M6_color)
    ax.plot(Temperatures-273.15,D_dict['D12A']*1e4,'--',color=M12_color)
    ax.plot(Temperatures-273.15,D_dict['D18A']*1e4,'--',color=M18_color)
    ax.plot(Temperatures-273.15,D_dict['D24']*1e4,'--',color=M24_color)
    ax.plot(Temperatures-273.15,D_dict['D30']*1e4,'--',color=M30_color)
    ax.plot(Temperatures-273.15,D_dict['D60']*1e4,'--',color=M60_color)

    fig.tight_layout()
    fig.savefig(f"{params['Figure name']}",format='pdf')
    
main()
    
    