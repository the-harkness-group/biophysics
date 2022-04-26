#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:25:50 2021

@author: robertharkness
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
import yaml
import customcolormap

def main():
    
    # Read yaml template to get DLS data filename, then read it
    params = yaml.safe_load(open(sys.argv[1],'r'))
    data =  pd.read_csv(params['Outname'])
    date = params['Date']
    
    # Plot DLS manifolds for each sample in the experiment/data file according to
    # the yaml template
    for sample in params['Samples'].keys():
        
        SampleData = data[data['Sample'] == sample].copy()        
        SampleData.Temperature = round(SampleData.Temperature*2)/2
        SampleGroups_byTemp = SampleData.groupby('Temperature')
        SampleGroups_byConc = SampleData.groupby('Concentration')
        
        plot_fit(SampleGroups_byTemp, SampleGroups_byConc, sample, date)

### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(DLSdata_groupedbyTemp, DLSdata_groupedbyConc, sample, date):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"DLSplots_{sample}_{date}.pdf")
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    #plt.rcParams['axes.linewidth'] = 2
    plt.style.use('figure')
    
    hex_colors = ['#020101','#281112','#4E2128','#733243','#984364','#BD548A','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF'] # Pink to black colormap
    hex_colors = list(reversed(hex_colors))

    cmap = customcolormap.get_continuous_cmap(hex_colors, DLSdata_groupedbyConc.ngroups) #  Get HEX color at specified number of points using the above HEX colors
    
    DT_fig, DT_ax = plt.subplots(1,1,figsize=(6.3,5.14))
    #DT_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    DT_ax.yaxis.major.formatter._useMathText = True
    DT_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    DT_ax.set_title(f"{sample}")
    DT_ax.set_xlabel('Temperature \N{DEGREE SIGN}C')
    DT_ax.set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$')
    #DT_ax.set_ylim([1e-7,8.5e-7])
    DT_ax.set_xticks([10,20,30,40,50])
    #DT_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))

    
    for ind, group in DLSdata_groupedbyTemp:

        Temperatures = np.array(group.Temperature)
        
        fig, axs = plt.subplots(1,2,figsize=(15,7))
        axs[0].plot(group.Concentration,group.D*1e4,'ko')
        axs[0].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        axs[0].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        axs[0].yaxis.major.formatter._useMathText = True
        axs[0].set_title(f"{sample}\nTemperature: {Temperatures[0]} \N{DEGREE SIGN}C")
        axs[0].set_xlabel('$[M]_T$ $\mu$M')
        axs[0].set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$')
        axs[0].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        
        axs[1].plot(group.Concentration,group.PD*100,'ko')
        axs[1].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        axs[1].yaxis.major.formatter._useMathText = True
        axs[1].set_title(f"{sample}\nTemperature: {Temperatures[0]} \N{DEGREE SIGN}C")
        axs[1].set_xlabel('$[M]_T$ $\mu$M')
        axs[1].set_ylabel('Polydispersity %')
        axs[1].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        
        fig.tight_layout()
        
        pdf.savefig(fig)
        plt.close(fig)
    
    cidx = 0
    for ind, group in DLSdata_groupedbyConc:
        
        DT_ax.plot(group.Temperature,group.D*1e4,'o',color=cmap[cidx])
        cidx += 1
    
    DT_fig.tight_layout()
    pdf.savefig(DT_fig)
    pdf.close()
    
main()