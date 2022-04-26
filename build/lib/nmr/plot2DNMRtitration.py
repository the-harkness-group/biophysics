#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:18:49 2021

@author: robertharkness
"""

import sys
import numpy as np
import nmrglue as ng
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import yaml
import customcolormap

def main():
    
    # Read in data to be plotted
    params = yaml.safe_load(open(sys.argv[1],'r'))
    
    # plot parameters
    #cmap = matplotlib.cm.Blues_r    # contour map (colors to use for contours)
    hex_colors = ['#130F2C','#2B1C4C','#4C296B','#74378A','#A346A8','#C556B3','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF'][::-1] # Black pink reversed
    #basecmap = customcolormap.get_continuous_cmap(hex_colors,len(params['Files']))
    #basecmap = cm.get_cmap('jet',4)
    #basecmap = ['#5F5AED','#BF60E7','#F9FF5C','#58FFE1']
    coolwarm = cm.get_cmap('Spectral',4) # 4 for S210A and S210A/N45F, only 2 for S210A/Y444A
    basecmap = coolwarm(np.linspace(0, 1, 4))
    #print(basecmap)
    #cmap = matplotlib.cm.BuPu_r
    #contour_start = [6e5,5e5,6e5,9e5] # contour level start value #L272 S210A
    #contour_start = [8e5,4e5,1.3e6,9e5] # contour level start value #L272 S210A/N45F
    #contour_start = [7e5,6e6] # contour start L272 S210A/Y444A
    #contour_start = [5e5,2e5,5e5,1.1e6] # V293 S210A
    contour_start = [6e5,6e5,6e5,6e5] # V293 PDZ1
    contour_num = 20            # number of contour levels
    contour_factor = 1.20          # scaling factor between contour levels
    plt.style.use('figure')
    # create the figure
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    
    # Chemical shift indices for peaks of interest
    L276_M3 = [0.724,24.130] # Trimer peaks
    L272_M3_1 = [0.730,24.627]
    L272_M3_2 = [0.704,25.098]
    V290_M3 = [0.629,21.236]
    V293_M3 = [0.570,22.326]
    V309_M3 = [0.814,21.468]
    
    L276_PDZ1_free = [0.737,24.130] # PDZ1 free peaks
    L272_PDZ1_1_free = [0.736,24.669]
    L272_PDZ1_2_free = [0.713,25.060]
    V290_PDZ1_free = [0.634,21.284]
    V293_PDZ1_free = [0.561,22.354]
    V309_PDZ1_free = [0.802,21.415]
    
    L276_PDZ1_bound = [0.413,24.935] # PDZ1 bound peaks
    L272_PDZ1_1_bound = [0.609,25.646]
    L272_PDZ1_2_bound = [0.668,24.064]
    V290_PDZ1_bound = [0.672,21.335]
    V293_PDZ1_bound = [0.654,22.428]
    V309_PDZ1_bound = [0.748,21.624]
    
    # Calculate S210A bound peaks based on PDZ1 bound peaks, shift according to offset of free trimer and free PDZ1
    L276_WT_bound = [L276_M3[0] + (L276_PDZ1_bound[0] - L276_PDZ1_free[0]) + (L276_PDZ1_free[0] - L276_M3[0]),
                     L276_M3[1] + (L276_PDZ1_bound[1] - L276_PDZ1_free[1]) + (L276_PDZ1_free[1] - L276_M3[1])]
    L272_WT_1_bound = [L272_M3_1[0] + (L272_PDZ1_1_bound[0] - L272_PDZ1_1_free[0]) + (L272_PDZ1_1_free[0] - L272_M3_1[0]),
                     L272_M3_1[1] + (L272_PDZ1_1_bound[1] - L272_PDZ1_1_free[1]) + (L272_PDZ1_1_free[1] - L272_M3_1[1])]
    L272_WT_2_bound = [L272_M3_2[0] + (L272_PDZ1_2_bound[0] - L272_PDZ1_2_free[0]) + (L272_PDZ1_2_free[0] - L272_M3_2[0]),
                     L272_M3_2[1] + (L272_PDZ1_2_bound[1] - L272_PDZ1_2_free[1]) + (L272_PDZ1_2_free[1] - L272_M3_2[1])]
    V290_WT_bound = [V290_M3[0] + (V290_PDZ1_bound[0] - V290_PDZ1_free[0]) + (V290_PDZ1_free[0] - V290_M3[0]),
                     V290_M3[1] + (V290_PDZ1_bound[1] - V290_PDZ1_free[1]) + (V290_PDZ1_free[1] - V290_M3[1])]
    V293_WT_bound = [V293_M3[0] + (V293_PDZ1_bound[0] - V293_PDZ1_free[0]) + (V293_PDZ1_free[0] - V293_M3[0]),
                     V293_M3[1] + (V293_PDZ1_bound[1] - V293_PDZ1_free[1]) + (V293_PDZ1_free[1] - V293_M3[1])]
    V309_WT_bound = [V309_M3[0] + (V309_PDZ1_bound[0] - V309_PDZ1_free[0]) + (V309_PDZ1_free[0] - V309_M3[0]),
                     V309_M3[1] + (V309_PDZ1_bound[1] - V309_PDZ1_free[1]) + (V309_PDZ1_free[1] - V309_M3[1])]
    
    # Calculate lines between the free and bound shifts for S210A
    L276_WT_m = (L276_WT_bound[1] - L276_M3[1])/(L276_WT_bound[0] - L276_M3[0])
    L276_WT_b = L276_M3[1] - L276_WT_m*L276_M3[0]
    L276_WT_line = L276_WT_m*np.linspace(L276_WT_bound[0],L276_M3[0],15) + L276_WT_b
    
    L272_WT_1_m = (L272_WT_1_bound[1] - L272_M3_1[1])/(L272_WT_1_bound[0] - L272_M3_1[0])
    L272_WT_1_b = L272_M3_1[1] - L272_WT_1_m*L272_M3_1[0]
    L272_WT_1_line = L272_WT_1_m*np.linspace(L272_WT_1_bound[0],L272_M3_1[0],15) + L272_WT_1_b
    
    L272_WT_2_m = (L272_WT_2_bound[1] - L272_M3_2[1])/(L272_WT_2_bound[0] - L272_M3_2[0])
    L272_WT_2_b = L272_M3_2[1] - L272_WT_2_m*L272_M3_2[0]
    L272_WT_2_line = L272_WT_2_m*np.linspace(L272_WT_2_bound[0],L272_M3_2[0],15) + L272_WT_2_b
    
    V290_WT_m = (V290_WT_bound[1] - V290_M3[1])/(V290_WT_bound[0] - V290_M3[0])
    V290_WT_b = V290_M3[1] - V290_WT_m*V290_M3[0]
    V290_WT_line = V290_WT_m*np.linspace(V290_WT_bound[0],V290_M3[0],7) + V290_WT_b
    
    V293_WT_m = (V293_WT_bound[1] - V293_M3[1])/(V293_WT_bound[0] - V293_M3[0])
    V293_WT_b = V293_M3[1] - V293_WT_m*V293_M3[0]
    V293_WT_line = V293_WT_m*np.linspace(V293_WT_bound[0],V293_M3[0],7) + V293_WT_b
    
    V309_WT_m = (V309_WT_bound[1] - V309_M3[1])/(V309_WT_bound[0] - V309_M3[0])
    V309_WT_b = V309_M3[1] - V309_WT_m*V309_M3[0]
    V309_WT_line = V309_WT_m*np.linspace(V309_WT_bound[0],V309_M3[0],7) + V309_WT_b
    
    # Calculate lines between free and bound for PDZ1
    L276_PDZ1_m = (L276_PDZ1_bound[1] - L276_PDZ1_free[1])/(L276_PDZ1_bound[0] - L276_PDZ1_free[0])
    L276_PDZ1_b = L276_PDZ1_free[1] - L276_PDZ1_m*L276_PDZ1_free[0]
    L276_PDZ1_line = L276_PDZ1_m*np.linspace(L276_PDZ1_bound[0],L276_PDZ1_free[0],15) + L276_PDZ1_b
    
    L272_PDZ1_1_m = (L272_PDZ1_1_bound[1] - L272_PDZ1_1_free[1])/(L272_PDZ1_1_bound[0] - L272_PDZ1_1_free[0])
    L272_PDZ1_1_b = L272_PDZ1_1_free[1] - L272_PDZ1_1_m*L272_PDZ1_1_free[0]
    L272_PDZ1_1_line = L272_PDZ1_1_m*np.linspace(L272_PDZ1_1_bound[0],L272_PDZ1_1_free[0],15) + L272_PDZ1_1_b
    
    L272_PDZ1_2_m = (L272_PDZ1_2_bound[1] - L272_PDZ1_2_free[1])/(L272_PDZ1_2_bound[0] - L272_PDZ1_2_free[0])
    L272_PDZ1_2_b = L272_PDZ1_2_free[1] - L272_PDZ1_2_m*L272_PDZ1_2_free[0]
    L272_PDZ1_2_line = L272_PDZ1_2_m*np.linspace(L272_PDZ1_2_bound[0],L272_PDZ1_2_free[0],15) + L272_PDZ1_2_b
    
    V290_PDZ1_m = (V290_PDZ1_bound[1] - V290_PDZ1_free[1])/(V290_PDZ1_bound[0] - V290_PDZ1_free[0])
    V290_PDZ1_b = V290_PDZ1_free[1] - V290_PDZ1_m*V290_PDZ1_free[0]
    V290_PDZ1_line = V290_PDZ1_m*np.linspace(V290_PDZ1_bound[0],V290_PDZ1_free[0],7) + V290_PDZ1_b
    
    V293_PDZ1_m = (V293_PDZ1_bound[1] - V293_PDZ1_free[1])/(V293_PDZ1_bound[0] - V293_PDZ1_free[0])
    V293_PDZ1_b = V293_PDZ1_free[1] - V293_PDZ1_m*V293_PDZ1_free[0]
    V293_PDZ1_line = V293_PDZ1_m*np.linspace(V293_PDZ1_bound[0],V293_PDZ1_free[0],7) + V293_PDZ1_b
    
    V309_PDZ1_m = (V309_PDZ1_bound[1] - V309_PDZ1_free[1])/(V309_PDZ1_bound[0] - V309_PDZ1_free[0])
    V309_PDZ1_b = V309_PDZ1_free[1] - V309_PDZ1_m*V309_PDZ1_free[0]
    V309_PDZ1_line = V309_PDZ1_m*np.linspace(V309_PDZ1_bound[0],V309_PDZ1_free[0],7) + V309_PDZ1_b
    
    
    #print((L276_WT_bound[1] - L276_M3[1])/(L276_WT_bound[0] - L276_M3[0]))
    #L272_WT_1_bound = []
    
    # calculate contour levels
    #cl = contour_start * contour_factor ** np.arange(contour_num) 
    
    data_dict = {f"{file}":{} for file in params['Files']}
    for index, file in enumerate(params['Files']):
        data_dict[file]['dic, data'] = ng.sparky.read(file)

        # Make ppm scales
        data_dict[file]['Direct uc'] = ng.sparky.make_uc(data_dict[file]['dic, data'][0], data_dict[file]['dic, data'][1], dim=1) # Direct dimension
        data_dict[file]['Direct ppm scale'] = data_dict[file]['Direct uc'].ppm_scale()
        data_dict[file]['Direct ppm limits'] = data_dict[file]['Direct uc'].ppm_limits()
        
        data_dict[file]['Indirect uc'] = ng.sparky.make_uc(data_dict[file]['dic, data'][0], data_dict[file]['dic, data'][1], dim=0) # Indirect dimension
        data_dict[file]['Indirect ppm scale'] = data_dict[file]['Indirect uc'].ppm_scale()
        data_dict[file]['Indirect ppm limits'] = data_dict[file]['Indirect uc'].ppm_limits()
        
        if index == 0:
            
            cl = contour_start[index] * contour_factor ** np.arange(contour_num)
            
            # plot the contours
            ax.contour(data_dict[file]['dic, data'][1], cl, colors=[tuple(basecmap[index])], 
                       extent=(data_dict[file]['Direct ppm limits'][0], data_dict[file]['Direct ppm limits'][1], data_dict[file]['Indirect ppm limits'][0], data_dict[file]['Indirect ppm limits'][1]))
            
        if index != 0:
            
            cl = contour_start[index] * contour_factor ** np.arange(contour_num)
            
            # plot the contours
            ax.contour(data_dict[file]['dic, data'][1], cl, colors=[tuple(basecmap[index])], 
                       extent=(data_dict[file]['Direct ppm limits'][0], data_dict[file]['Direct ppm limits'][1], data_dict[file]['Indirect ppm limits'][0], data_dict[file]['Indirect ppm limits'][1]))
    
    # Plot peak markers and lines connecting free and bound
    # S210A
#    ax.plot(np.linspace(L276_WT_bound[0],L276_M3[0],15),L276_WT_line,'k.',linewidth=2)
#    ax.plot(L276_M3[0],L276_M3[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
#    ax.plot(L276_WT_bound[0],L276_WT_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
#    
#    ax.plot(np.linspace(L272_WT_1_bound[0],L272_M3_1[0],15),L272_WT_1_line,'k.',linewidth=2)
#    ax.plot(L272_M3_1[0],L272_M3_1[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
#    ax.plot(L272_WT_1_bound[0],L272_WT_1_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
#    
#    ax.plot(np.linspace(L272_WT_2_bound[0],L272_M3_2[0],15),L272_WT_2_line,'k.',linewidth=2)
#    ax.plot(L272_M3_2[0],L272_M3_2[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
#    ax.plot(L272_WT_2_bound[0],L272_WT_2_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
#    
#    ax.plot(np.linspace(V290_WT_bound[0],V290_M3[0],7),V290_WT_line,'k.',linewidth=2)
#    ax.plot(V290_M3[0],V290_M3[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
#    ax.plot(V290_WT_bound[0],V290_WT_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
#    
#    ax.plot(np.linspace(V293_WT_bound[0],V293_M3[0],7),V293_WT_line,'k.',linewidth=2)
#    ax.plot(V293_M3[0],V293_M3[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
#    ax.plot(V293_WT_bound[0],V293_WT_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
#    
#    ax.plot(np.linspace(V309_WT_bound[0],V309_M3[0],7),V309_WT_line,'k.',linewidth=2)
#    ax.plot(V309_M3[0],V309_M3[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
#    ax.plot(V309_WT_bound[0],V309_WT_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
    
    # PDZ1
    ax.plot(np.linspace(L276_PDZ1_bound[0],L276_PDZ1_free[0],15),L276_PDZ1_line,'k.',linewidth=2)
    ax.plot(L276_PDZ1_free[0],L276_PDZ1_free[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
    ax.plot(L276_PDZ1_bound[0],L276_PDZ1_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
    
    ax.plot(np.linspace(L272_PDZ1_1_bound[0],L272_PDZ1_1_free[0],15),L272_PDZ1_1_line,'k.',linewidth=2)
    ax.plot(L272_PDZ1_1_free[0],L272_PDZ1_1_free[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
    ax.plot(L272_PDZ1_1_bound[0],L272_PDZ1_1_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
    
    ax.plot(np.linspace(L272_PDZ1_2_bound[0],L272_PDZ1_2_free[0],15),L272_PDZ1_2_line,'k.',linewidth=2)
    ax.plot(L272_PDZ1_2_free[0],L272_PDZ1_2_free[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
    ax.plot(L272_PDZ1_2_bound[0],L272_PDZ1_2_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
    
    ax.plot(np.linspace(V290_PDZ1_bound[0],V290_PDZ1_free[0],7),V290_PDZ1_line,'k.',linewidth=2)
    ax.plot(V290_PDZ1_free[0],V290_PDZ1_free[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
    ax.plot(V290_PDZ1_bound[0],V290_PDZ1_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
    
    ax.plot(np.linspace(V293_PDZ1_bound[0],V293_PDZ1_free[0],7),V293_PDZ1_line,'k.',linewidth=2)
    ax.plot(V293_PDZ1_free[0],V293_PDZ1_free[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
    ax.plot(V293_PDZ1_bound[0],V293_PDZ1_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
    
    ax.plot(np.linspace(V309_PDZ1_bound[0],V309_PDZ1_free[0],7),V309_PDZ1_line,'k.',linewidth=2)
    ax.plot(V309_PDZ1_free[0],V309_PDZ1_free[1],'k*',markersize=15,markerfacecolor='w',markeredgecolor='k',linewidth=2)
    ax.plot(V309_PDZ1_bound[0],V309_PDZ1_bound[1],'k*',markersize=15,markerfacecolor='k',markeredgecolor='k',linewidth=2)
        
    # decorate the axes
    ax.set_ylabel(f"{params['Indirect dimension']} ppm")
    ax.set_xlabel(f"{params['Direct dimension']} ppm")
    ax.set_xlim(params['Direct limits'])
    ax.set_ylim(params['Indirect limits'])
    ax.set_yticks([22.5,22.0,21.5,21.0])

    # save the figure
    fig.tight_layout()
    fig.savefig(f"{params['Figure filename']}",format='pdf')

    # plot slices in each direction
    #xslice = data[uc_15n("111.27 ppm"), :]
    #ax.plot(ppm_13c, -xslice / 4.e4 + 111.27)
    #yslice = data[:, uc_13c("62.0 ppm")]
    #ax.plot(yslice / 2.e4 + 62.0, ppm_15n)

main()