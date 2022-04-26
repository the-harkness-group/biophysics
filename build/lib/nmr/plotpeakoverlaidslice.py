#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 16:15:46 2021

@author: robertharkness
"""

import sys
import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
from matplotlib import cm
import yaml
import pandas as pd
#import customcolormap

def main():
    
    # Read in data to be plotted
    params = yaml.safe_load(open(sys.argv[1],'r'))
    
    # Plot parameters
    if 'Color space' in params.keys():
        cm_space = np.array(params['Color space'])
    else:
        cm_space = np.linspace(0,1,len(params['Sparky files']))
    basecmap = [cm.coolwarm(x) for x in cm_space][::-1]
    finalcolor = basecmap[-1]
    basecmap = [finalcolor for x in basecmap]
    
    #hex_colors = ['#020101','#281112','#4E2128','#733243','#984364','#BD548A','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF'] # Colors
    #hex_colors = list(reversed(hex_colors))
    #cm_space = [float(cm_space[x]) for x in range(len(cm_space))]
    #basecmap = customcolormap.get_continuous_rgba_cmap(hex_colors,len(params['Sparky files']),cm_space)
    plt.style.use('figure')
    
    # Plot a peak of interest from a desired view in the NMR data
    # Run through planes of a titration
    for peak in params['Single peaks']:
        
        # Get peak list
        extras_dict = {'Free peak':[],'Bound peak':[],'Line':[],'Ref_w2':[],'Ref_w1':[]}
        peaklist = pd.read_csv(params['Peak list'],delim_whitespace=True) # Peak list
        peaklist.columns = ['Assignment','w1','w2','w1 Hz','w2 Hz','dummy','dummy']

        # Get contour parameters for peak
        contour_start = params['Single peaks'][peak]['Contour start']
        contour_factor = params['Single peaks'][peak]['Factor']
        contour_num = params['Single peaks'][peak]['Levels']
        cl = float(contour_start) * float(contour_factor) ** np.arange(float(contour_num))
        
        for index, file in enumerate(params['Sparky files']):

            # Set up figure
            fig, ax = plt.subplots(1,1,figsize=(11,5.5))

            # Read data
            dic, data = ng.sparky.read(file) # Spectrum

            # Make ppm scales
            direct_uc = ng.sparky.make_uc(dic, data, dim=1) # Direct dimension
            direct_ppm = direct_uc.ppm_scale()
            direct_limits = direct_uc.ppm_limits()
        
            indirect_uc = ng.sparky.make_uc(dic, data, dim=0) # Indirect dimension
            indirect_ppm = indirect_uc.ppm_scale()
            indirect_limits = indirect_uc.ppm_limits()
            
            # Get data box around peak max +/- certain padding in indirect and direct dimensions
            # Get ppm range for peakbox
            direct_coordinates = peaklist[peaklist['Assignment'] == peak]['w2'].values
            indirect_coordinates = peaklist[peaklist['Assignment'] == peak]['w1'].values
            
            if index == 0: # Get free peak coordinates
                extras_dict['Free peak'].append((direct_coordinates,indirect_coordinates))
            
                if 'Referencing' in params.keys():
                    if params['Referencing'] == 'Internal':
                        extras_dict['Ref_w2'] =  peaklist[peaklist['Assignment'] == 'IRC1-H1']['w2'].values
                        extras_dict['Ref_w1'] =  peaklist[peaklist['Assignment'] == 'IRC1-H1']['w1'].values
            
            peakbox_directppmindices = [(np.abs(direct_ppm - (direct_coordinates - params['Single peaks'][peak]['Direct pad']))).argmin(),
            (np.abs(direct_ppm - (direct_coordinates + params['Single peaks'][peak]['Direct pad']))).argmin()]
            
            peakbox_indirectppmindices = [(np.abs(indirect_ppm - (indirect_coordinates - params['Single peaks'][peak]['Indirect pad']))).argmin(),
            (np.abs(indirect_ppm - (indirect_coordinates + params['Single peaks'][peak]['Indirect pad']))).argmin()]
            
            # Get peakbox using indices
            peakbox = data[peakbox_indirectppmindices[-1]:peakbox_indirectppmindices[0]+1,
                peakbox_directppmindices[-1]:peakbox_directppmindices[0]+1]

            # Get indirect dimension slice to plot overlaid with peak contours
            peakslice = data[indirect_uc(f"{params['Single peaks'][peak]['Slice ppm']} ppm"), :]
            
            # Shift ppm indices if referencing indicated
            if 'Referencing' in params.keys():
                if params['Referencing'] == 'Internal':
                    direct_ppm = direct_ppm + extras_dict['Ref_w2'] - peaklist[peaklist['Assignment'] == 'IRC1-H1']['w2'].values
                    indirect_ppm = indirect_ppm + extras_dict['Ref_w1'] - peaklist[peaklist['Assignment'] == 'IRC1-H1']['w1'].values
         
            ax.contour(peakbox, cl, colors=[basecmap[index]], linewidths=2, # 2D data overlaid with direct lineshape slices, plot only peak in specified ppm box
                   extent=(direct_ppm[peakbox_directppmindices[-1]], direct_ppm[peakbox_directppmindices[0]], 
                           indirect_ppm[peakbox_indirectppmindices[-1]], indirect_ppm[peakbox_indirectppmindices[0]]))

            # Plot slice overlaid with contour plot
            ax.plot(direct_ppm,(-peakslice/float(params['Single peaks'][peak]['Slice scale'])+float(params['Single peaks'][peak]['Slice ppm'])
            +float(params['Single peaks'][peak]['Slice shift']))
            ,color=basecmap[index])
        
            # Add markers for positions of interest
            ax.plot(params['Single peaks'][peak]['Bound peak'][0],params['Single peaks'][peak]['Bound peak'][1]
            ,'*',markerfacecolor='k',markeredgecolor='k',markersize=20)
            #ax.plot(extras_dict['Free peak'][0][0],extras_dict['Free peak'][0][1],'*',markerfacecolor='#fde0ef',markeredgecolor='k',markeredgewidth=2,markersize=25)
        
            # Decorate the axes
            ax.set_ylabel(f"{params['Indirect dimension']} ppm")
            ax.set_xlabel(f"{params['Direct dimension']} ppm")
            ax.set_xlim(params['Single peaks'][peak]['Direct limits'])
            ax.set_ylim(params['Single peaks'][peak]['Indirect limits'])
            if ('xticks' in params['Single peaks'][peak].keys()):
                ax.set_xticks(params['Single peaks'][peak]['xticks'])
            if ('yticks' in params['Single peaks'][peak].keys()):
                ax.set_yticks(params['Single peaks'][peak]['yticks'])

            # Save the figure
            fig.tight_layout()
            fig.savefig(f"{peak}_{params['Concentrations'][index]}uM.pdf",format='pdf')
        
main()