#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:11:29 2020

@author: robertharkness
"""

import sys
import numpy as np
import nmrglue as ng
import matplotlib
import matplotlib.pyplot as plt
import yaml
import customcolormap

def main():
    
    # Read in data to be plotted
    params = yaml.safe_load(open(sys.argv[1],'r'))
    
    # plot parameters
    #cmap = matplotlib.cm.Blues_r    # contour map (colors to use for contours)
    hex_colors = ['#130F2C','#2B1C4C','#4C296B','#74378A','#A346A8','#C556B3','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF'][::-1] # Black pink reversed
    basecmap = customcolormap.get_continuous_cmap(hex_colors,len(params['Files']))
    #cmap = matplotlib.cm.BuPu_r
    contour_start = 2.5e5           # contour level start value
    contour_num = 20              # number of contour levels
    contour_factor = 1.20          # scaling factor between contour levels
    plt.style.use('figure')

    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num) 
    
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

        # create the figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # plot the contours
        ax.contour(data_dict[file]['dic, data'][1], cl, color=basecmap[index], 
                   extent=(data_dict[file]['Direct ppm limits'][0], data_dict[file]['Direct ppm limits'][1], data_dict[file]['Indirect ppm limits'][0], data_dict[file]['Indirect ppm limits'][1]))
        
        # decorate the axes
        ax.set_ylabel(f"{params['Indirect dimension']} ppm")
        ax.set_xlabel(f"{params['Direct dimension']} ppm")
        ax.set_xlim(params['Direct limits'])
        ax.set_ylim(params['Indirect limits'])

        # save the figure
        fig.tight_layout()
        fig.savefig(f"{file}.png")
        plt.show()
    

    # plot slices in each direction
    #xslice = data[uc_15n("111.27 ppm"), :]
    #ax.plot(ppm_13c, -xslice / 4.e4 + 111.27)
    #yslice = data[:, uc_13c("62.0 ppm")]
    #ax.plot(yslice / 2.e4 + 62.0, ppm_15n)

main()