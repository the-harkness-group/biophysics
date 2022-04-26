#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:42:18 2020

@author: robertharkness
"""

import sys
import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
import yaml
import matplotlib.gridspec as gridspec

def main():
    
    # Read in parameters for data to be plotted
    params = yaml.safe_load(open(sys.argv[1],'r'))
    
    simdata_dict, exptdata_dict = processdata(params) # Process data
    plotting(params, simdata_dict, exptdata_dict) # Plot processed data
          
def processdata(params):
    
    simdata_dict = {f"{file}":{} for file in params['Simulated Sparky Files']} # TITAN fits or otherwise
    exptdata_dict = {f"{file}":{} for file in params['Experiment Sparky Files']} # Experiment data
    for simfile, exptfile in zip(params['Simulated Sparky Files'],params['Experiment Sparky Files']):
        
        # Read in nmrglue data for simulated data
        simdata_dict[simfile]['dic, data'] = ng.sparky.read(simfile)

        # Make ppm scales
        simdata_dict[simfile]['Direct uc'] = ng.sparky.make_uc(simdata_dict[simfile]['dic, data'][0], simdata_dict[simfile]['dic, data'][1], dim=1) # Direct dimension
        simdata_dict[simfile]['Direct ppm scale'] = simdata_dict[simfile]['Direct uc'].ppm_scale()
        simdata_dict[simfile]['Direct ppm limits'] = simdata_dict[simfile]['Direct uc'].ppm_limits()
        
        simdata_dict[simfile]['Indirect uc'] = ng.sparky.make_uc(simdata_dict[simfile]['dic, data'][0], simdata_dict[simfile]['dic, data'][1], dim=0) # Indirect dimension
        simdata_dict[simfile]['Indirect ppm scale'] = simdata_dict[simfile]['Indirect uc'].ppm_scale()
        simdata_dict[simfile]['Indirect ppm limits'] = simdata_dict[simfile]['Indirect uc'].ppm_limits()
        
        # Read in nmrglue data for experimental data
        exptdata_dict[exptfile]['dic, data'] = ng.sparky.read(exptfile)

        # Make ppm scales
        exptdata_dict[exptfile]['Direct uc'] = ng.sparky.make_uc(exptdata_dict[exptfile]['dic, data'][0], exptdata_dict[exptfile]['dic, data'][1], dim=1) # Direct dimension
        exptdata_dict[exptfile]['Direct ppm scale'] = exptdata_dict[exptfile]['Direct uc'].ppm_scale()
        exptdata_dict[exptfile]['Direct ppm limits'] = exptdata_dict[exptfile]['Direct uc'].ppm_limits()
        
        exptdata_dict[exptfile]['Indirect uc'] = ng.sparky.make_uc(exptdata_dict[exptfile]['dic, data'][0], exptdata_dict[exptfile]['dic, data'][1], dim=0) # Indirect dimension
        exptdata_dict[exptfile]['Indirect ppm scale'] = exptdata_dict[exptfile]['Indirect uc'].ppm_scale()
        exptdata_dict[exptfile]['Indirect ppm limits'] = exptdata_dict[exptfile]['Indirect uc'].ppm_limits()
        
    return simdata_dict, exptdata_dict
        
def plotting(params, simdata_dict, exptdata_dict):
    
    # plot parameters
    basecmap = ['#130F2C','#2B1C4C','#4C296B','#74378A','#A346A8','#C556B3','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF'][::-1] # Black pink reversed        
    contour_start = float(params['Contour start'])      # contour level start value
    contour_num = 20              # number of contour levels
    contour_factor = 1.20          # scaling factor between contour levels
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.style.use('figure')
    cl = contour_start * contour_factor ** np.arange(contour_num) # contour levels
    
    if not params['Direct coordinates'] or params.get('Direct coordinates') == None:
        for index, simfile in enumerate(params['Simulated Sparky Files']):
        
            # Make a figure
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # plot the contours
            ax.contour(simdata_dict[simfile]['dic, data'][1], cl, colors=basecmap[index], 
                   extent=(simdata_dict[simfile]['Direct ppm limits'][0], simdata_dict[simfile]['Direct ppm limits'][1], simdata_dict[simfile]['Indirect ppm limits'][0], simdata_dict[simfile]['Indirect ppm limits'][1]))
            # decorate the axes
            ax.set_title("Write down the peak coordinates",fontsize=20)
            ax.set_ylabel(f"{params['Indirect dimension']} ppm")
            ax.set_xlabel(f"{params['Direct dimension']} ppm")
            ax.set_xlim(params['Direct limits'])
            ax.set_ylim(params['Indirect limits'])
            plt.show()
    
    else:
        # Make figures
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)
        fig3, ax3 = plt.subplots(1,1,figsize=(9,6.5))
        
        late_offset = 0.9 # N273 50C
        early_offset = 0.8 # N273 50C
        #late_offset = -1.1 # A284 25C
        #early_offset = -0.2 # A284 25C
        for index, (simfile, exptfile) in enumerate(zip(params['Simulated Sparky Files'], params['Experiment Sparky Files'])):

            # plot the 1D slices for each dimension
            xsimslice = simdata_dict[simfile]['dic, data'][1][simdata_dict[simfile]['Indirect uc'](str(params['Indirect coordinates'][index]) + ' ppm'),:]
            ysimslice = simdata_dict[simfile]['dic, data'][1][:,simdata_dict[simfile]['Direct uc'](str(params['Direct coordinates'][index]) + ' ppm')]
            
            xexptslice = exptdata_dict[exptfile]['dic, data'][1][exptdata_dict[exptfile]['Indirect uc'](str(params['Indirect coordinates'][index]) + ' ppm'),:]
            yexptslice = exptdata_dict[exptfile]['dic, data'][1][:,exptdata_dict[exptfile]['Direct uc'](str(params['Direct coordinates'][index]) + ' ppm')]
            
            xsimslice_ppmindices = np.where((simdata_dict[simfile]['Direct ppm scale'] >= params['Direct coordinates'][index] - params['Direct pad']) 
            & (simdata_dict[simfile]['Direct ppm scale'] <= params['Direct coordinates'][index] + params['Direct pad']))
            xsimslice_ppmrange = simdata_dict[simfile]['Direct ppm scale'][xsimslice_ppmindices]
            xsimslice_peak = xsimslice[xsimslice_ppmindices]
            
            xexptslice_ppmindices = np.where((exptdata_dict[exptfile]['Direct ppm scale'] >= params['Direct coordinates'][index] - params['Direct pad']) 
            & (exptdata_dict[exptfile]['Direct ppm scale'] <= params['Direct coordinates'][index] + params['Direct pad']))
            xexptslice_ppmrange = exptdata_dict[exptfile]['Direct ppm scale'][xexptslice_ppmindices]
            xexptslice_peak = xexptslice[xexptslice_ppmindices]
            
            # Get data box around peak max +/- certain padding in indirect and direct dimensions
            # Get ppm range for peakbox
            peakbox_directppmindices = [(np.abs(exptdata_dict[exptfile]['Direct ppm scale'] - (params['Direct coordinates'][index] - params['Direct pad']))).argmin(),
            (np.abs(exptdata_dict[exptfile]['Direct ppm scale'] - (params['Direct coordinates'][index] + params['Direct pad']))).argmin()]
            
            peakbox_indirectppmindices = [(np.abs(exptdata_dict[exptfile]['Indirect ppm scale'] - (params['Indirect coordinates'][index] - params['Indirect pad']))).argmin(),
            (np.abs(exptdata_dict[exptfile]['Indirect ppm scale'] - (params['Indirect coordinates'][index] + params['Indirect pad']))).argmin()]
            
            # Get peakbox using indices
            peakbox = exptdata_dict[exptfile]['dic, data'][1][peakbox_indirectppmindices[-1]:peakbox_indirectppmindices[0]+1,
                peakbox_directppmindices[-1]:peakbox_directppmindices[0]+1]
            
#            # Get ppm ranges for slices, was just checking if this produces identical behavior to the above section for getting slices
#            xsimslice_ppmindices = [(np.abs(exptdata_dict[exptfile]['Indirect ppm scale'] - params['Indirect coordinates'][index])).argmin()]
#            xexptslice_ppmindices = [(np.abs(exptdata_dict[exptfile]['Indirect ppm scale'] - params['Indirect coordinates'][index])).argmin()]
#
#            simindices = [(np.abs(simdata_dict[simfile]['Direct ppm scale'] - (params['Direct coordinates'][index] - params['Direct pad']))).argmin(),
#            (np.abs(simdata_dict[simfile]['Direct ppm scale'] - (params['Direct coordinates'][index] + params['Direct pad']))).argmin()]            
#            exptindices = [(np.abs(exptdata_dict[exptfile]['Direct ppm scale'] - (params['Direct coordinates'][index] - params['Direct pad']))).argmin(),
#            (np.abs(exptdata_dict[exptfile]['Direct ppm scale'] - (params['Direct coordinates'][index] + params['Direct pad']))).argmin()]
#            
#            xsimslice_ppmrange = simdata_dict[simfile]['Direct ppm scale'][simindices[-1]:simindices[0]+1]
#            xexptslice_ppmrange = exptdata_dict[exptfile]['Direct ppm scale'][exptindices[-1]:exptindices[0]+1]
#            
#            # Get slice data
#            xsimslice_peak = simdata_dict[simfile]['dic, data'][1][xsimslice_ppmindices[0],simindices[-1]:simindices[0]+1]
#            xexptslice_peak = exptdata_dict[exptfile]['dic, data'][1][xexptslice_ppmindices[0],exptindices[-1]:exptindices[0]+1]
            
            ax1.plot(exptdata_dict[exptfile]['Direct ppm scale'],xexptslice,'o',markersize=6,color=basecmap[index]) # Direct lineshapes
            ax1.plot(simdata_dict[simfile]['Direct ppm scale'],xsimslice,linewidth=2,color=basecmap[index])
            ax2.plot(exptdata_dict[exptfile]['Indirect ppm scale'],yexptslice,'o',markersize=6,color=basecmap[index]) # Indirect lineshapes
            ax2.plot(simdata_dict[simfile]['Indirect ppm scale'],ysimslice,linewidth=2,color=basecmap[index])
            
            #ax3.contour(exptdata_dict[exptfile]['dic, data'][1], cl, colors=basecmap[index], # 2D data overlaid with direct lineshape slices
            #      extent=(exptdata_dict[exptfile]['Direct ppm limits'][0], exptdata_dict[exptfile]['Direct ppm limits'][1], exptdata_dict[exptfile]['Indirect ppm limits'][0], exptdata_dict[exptfile]['Indirect ppm limits'][1]))
            ax3.contour(peakbox, cl, colors=basecmap[index], # 2D data overlaid with direct lineshape slices, plot only peak in specified ppm box
                   extent=(exptdata_dict[exptfile]['Direct ppm scale'][peakbox_directppmindices[-1]], exptdata_dict[exptfile]['Direct ppm scale'][peakbox_directppmindices[0]], 
                           exptdata_dict[exptfile]['Indirect ppm scale'][peakbox_indirectppmindices[-1]], exptdata_dict[exptfile]['Indirect ppm scale'][peakbox_indirectppmindices[0]]))

            if (index < len(params['Simulated Sparky Files'])/2) & (np.mod(index,2) == 0): # Plot every second trace
            
                ax3.plot(xexptslice_ppmrange,(-xexptslice_peak/(12*float(params['Contour start']))) + params['Indirect coordinates'][index] + early_offset,'o',markersize=6,color=basecmap[index])
                ax3.plot(xsimslice_ppmrange,(-xsimslice_peak/(12*float(params['Contour start']))) + params['Indirect coordinates'][index] + early_offset,linewidth=2,color=basecmap[index])
                early_offset -= 0.05
                
            if (index >= len(params['Simulated Sparky Files'])/2) & (np.mod(index,2) == 0): # Plot every second trace
                
                # N273 scaling factor is 5.2*Contour start
                # A284 scaling factor is 12*Contour start
                
                ax3.plot(xexptslice_ppmrange,(-xexptslice_peak/(5.2*float(params['Contour start']))) + params['Indirect coordinates'][index] - late_offset,'o',markersize=6,color=basecmap[index])
                ax3.plot(xsimslice_ppmrange,(-xsimslice_peak/(5.2*float(params['Contour start']))) + params['Indirect coordinates'][index] - late_offset,linewidth=2,color=basecmap[index])
                late_offset -= 0.2 # N273 50C
                #late_offset += 0.2 # A284 25C
                
            # decorate the axes
            ax1.set_title(f"{params['Peak of interest']} {params['Temperature']} \N{DEGREE SIGN}C",fontsize=20,color=params['Plot color'])
            ax2.set_title(f"{params['Peak of interest']} {params['Temperature']} \N{DEGREE SIGN}C",fontsize=20,color=params['Plot color'])
            ax1.set_xlabel(f"{params['Direct dimension']} ppm")
            ax2.set_xlabel(f"{params['Indirect dimension']} ppm")
            ax1.set_xlim(params['Direct 1D limits'])
            ax1.set_ylim([-1e5,8e5])
            ax2.set_xlim(params['Indirect 1D limits'])
            ax1.set_ylabel('Intensity')
            ax2.set_ylabel('Intensity')
            ax3.set_xlim(params['Direct 1D limits'])
            ax3.set_ylim(params['Indirect 1D limits'])
            ax3.set_xlabel(f"{params['Direct dimension']} ppm")
            ax3.set_ylabel(f"{params['Indirect dimension']} ppm")
            ax3.set_title(f"{params['Peak of interest']} {params['Temperature']} \N{DEGREE SIGN}C",color=params['Plot color'])
            
            ax1.yaxis.major.formatter._useMathText = True
            ax2.yaxis.major.formatter._useMathText = True
            ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
            ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
       
        #ax1.text(0.3,0.95,f"$k_{{on}}$ = {params['kon']} $\pm$ {params['kon error']} $M^{{-1}}$ $s^{{-1}}$\n$k_{{off}}$ = {params['koff']} $\pm$ {params['koff error']} $s^{{-1}}$\n$K_{{D}}$ = {params['Kd']} $\pm$ {params['Kd error']} $\mu$M",fontsize=20,color=params['Plot color'],transform=ax1.transAxes,va="top")   
        #ax2.text(0.3,0.95,f"$k_{{on}}$ = {params['kon']} $\pm$ {params['kon error']} $M^{{-1}}$ $s^{{-1}}$\n$k_{{off}}$ = {params['koff']} $\pm$ {params['koff error']} $s^{{-1}}$\n$K_{{D}}$ = {params['Kd']} $\pm$ {params['Kd error']} $\mu$M",fontsize=20,color=params['Plot color'],transform=ax2.transAxes,va="top")   
        # A284
        #ax3.text(0.05,0.35,f"$k_{{on}}$ = {params['kon']} $\pm$ {params['kon error']} $M^{{-1}}$ $s^{{-1}}$\n$k_{{off}}$ = {params['koff']} $\pm$ {params['koff error']} $s^{{-1}}$\n$K_{{D}}$ = {params['Kd']} $\pm$ {params['Kd error']} $\mu$M",fontsize=16,color=params['Plot color'],transform=ax3.transAxes,va="top")
        ax3.text(0.3,0.95,f"$K_{{D}}$ = {params['Kd']} $\pm$ {params['Kd error']} $\mu$M",fontsize=24,color=params['Plot color'],transform=ax3.transAxes,va="top")
        # N273
        #ax3.text(0.37,0.99,f"$k_{{on}}$ = {params['kon']} $\pm$ {params['kon error']} $M^{{-1}}$ $s^{{-1}}$\n$k_{{off}}$ = {params['koff']} $\pm$ {params['koff error']} $s^{{-1}}$\n$K_{{D}}$ = {params['Kd']} $\pm$ {params['Kd error']} $\mu$M",fontsize=16,color=params['Plot color'],transform=ax3.transAxes,va="top")
        
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig1.savefig('directlinshapesandfits.pdf',format='pdf')
        fig2.savefig('indirectlineshapesandfits.pdf',format='pdf')
        fig3.savefig('2Doverlaidwithfits.pdf',format='pdf')
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        
main()