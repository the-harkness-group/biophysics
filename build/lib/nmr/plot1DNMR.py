#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:50:01 2020

@author: robertharkness
"""

import sys
import pandas as pd
import yaml
import numpy as np
import nmrglue as ng
import matplotlib
import matplotlib.pyplot as plt

def plot_1DNMR():
    
    # Read in filenames to plot, prepare dictionary for nmrglue processing
    params = yaml.safe_load(open(sys.argv[1],'r'))
    files = params['files']
    data_dic = {sample.split('.ft2')[0]:{'pdic':[],'pdata':[],'shiftdata':[],'udic':[],'uc':[],'ppm_scale':[],'ref_ppm_scale':[],'bl_shift':[],'dss_ppm':[]} for sample in files}
    
    # Call populations function for generating population-weighted average 1D 1H NMR spectra
    exch_dic = populations()
    
    # Set up plot params
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    #colors = ['#a6bddb','#74a9cf','#3690c0','#0570b0','#fa9fb5','#f768a1','#dd3497','#ae017e'] # pale to dark colors
    #colors = ['#a6bddb','#74a9cf','#3690c0','#045a8d','#fa9fb5','#f768a1','#dd3497','#7a0177'] # pale to dark, more intense blues and pinks
    #colors = ['#0570b0','#3690c0','#74a9cf','#a6bddb','#ae017e','#dd3497','#f768a1','#fa9fb5'] # dark to pale
    #colors = ['#045a8d','#3690c0','#74a9cf','#a6bddb','#7a0177','#dd3497','#f768a1','#fa9fb5'] # dark to pale, more intense blues and pinks
    #colors = ['#d73027','#fdae61','#abd9e9','#4575b4','#bf812d','#de77ae','#35978f','#1b7837'] # Reviewer colors
    colors = ['#4575b4','#abd9e9','#fdae61','#d73027','#bf812d','#de77ae','#35978f','#1b7837'] # Reviewer colors, order swapped for NMR figure
              
    #xlim = [14.4,10.7]
    #xlim = [6.8, 5.5] # sugars
    #xlim = [8.7, 7.2] # bases
    #xlim =[12.4,-0.1]
    xlim = [12.2,10.7] # iminos
    #xlim = [12.4, 10.6]
    ylim = [-1e6,1.9e7] # iminos
    #ylim = [-1e6,8e7] # bases
    #ylim = [-1e6,5e7] # sugars
    lw = 1.5
    offset = 0.62e7 # For shifting up spectra that are overlaid with pop. avg, imino region
    #offset = 3e7 # bases and sugars
    #offset = 3e7
    
    # Make plot figure and axes
    fig = plt.figure(figsize=(18,15))
    ax1 = plt.subplot2grid((5, 4), (4, 1), colspan=2)
    ax2 = plt.subplot2grid((5, 4), (0, 0), colspan=2)
    ax3 = plt.subplot2grid((5, 4), (1, 0), colspan=2)
    ax4 = plt.subplot2grid((5, 4), (2, 0), colspan=2)
    ax5 = plt.subplot2grid((5, 4), (3, 0), colspan=2)
    ax6 = plt.subplot2grid((5, 4), (0, 2), colspan=2)
    ax7 = plt.subplot2grid((5, 4), (1, 2), colspan=2)
    ax8 = plt.subplot2grid((5, 4), (2, 2), colspan=2)
    ax9 = plt.subplot2grid((5, 4), (3, 2), colspan=2)
    plt.tight_layout()
    ax_list = fig.axes
    
    # Iterate through 1D data files, generate Bruker dictionaries according to spectral parameters
    for n,f in enumerate(files):
        print('File: ',n,f)
        sample = f.split('.ft2')[0]
        data_dic[sample]['pdic'],data_dic[sample]['pdata'] = ng.pipe.read(f)
        data_dic[sample]['udic'] = ng.bruker.guess_udic(data_dic[sample]['pdic'], data_dic[sample]['pdata'])
        data_dic[sample]['udic'][0]['car'] = 4.77*500.302
        data_dic[sample]['udic'][0]['obs'] = 500.302
        data_dic[sample]['udic'][0]['sw'] = 10000
        data_dic[sample]['udic'][0]['label'] = '1H'
        data_dic[sample]['uc'] = ng.fileiobase.uc_from_udic(data_dic[sample]['udic'])
        data_dic[sample]['ppm_scale'] = data_dic[sample]['uc'].ppm_scale()
        
        # Find DSS ppm value at DSS peak max
        dss_ppmrange = np.where((data_dic[sample]['ppm_scale'] <= 0.03) & (data_dic[sample]['ppm_scale'] >= -0.03))
        dss_intrange = data_dic[sample]['pdata'][dss_ppmrange[0]]
        dss_index = dss_ppmrange[0][dss_intrange.argmax()]
        data_dic[sample]['dss_ppm'] = data_dic[sample]['ppm_scale'][dss_index]
        # Shift ppm scale so DSS is referenced to zero
        data_dic[sample]['ref_ppm_scale'] = data_dic[sample]['ppm_scale'] + abs(data_dic[sample]['dss_ppm'])
        
        # Shift each spectrum so baseline is roughly at zero intensity for better comparison
        data_dic[sample]['bl_shift'] = data_dic[sample]['pdata'][data_dic[sample]['uc']("13ppm")] # Picked 13ppm as flat baseline since no signals are here
        if data_dic[sample]['bl_shift'] > 0:
            data_dic[sample]['shiftdata'] = data_dic[sample]['pdata'] - data_dic[sample]['bl_shift']
        elif data_dic[sample]['bl_shift'] < 0:
            data_dic[sample]['shiftdata'] = data_dic[sample]['pdata'] + np.abs(data_dic[sample]['bl_shift'])
            
    # Plot data in big plot according to colors
    for file in files:
        sample = file.split('.ft2')[0]
        if sample == 'WTP':
            
            WT_offset = 2e7 # imino region
            #WT_offset = 1e8 # base region
            #WT_offset = 3e7 # sugar region
            ax1.plot(data_dic[sample]['ref_ppm_scale'],data_dic[sample]['shiftdata']+WT_offset,'k',linewidth=lw,label='')
            
            # Population-weighted average using 3X = 14 and 5X = 11
            threeex_fiveex_avg = (exch_dic['14']['Q']/exch_dic['WT']['Q'])*np.array(data_dic['14P']['shiftdata']) + (exch_dic['11']['Q']/exch_dic['WT']['Q'])*np.array(data_dic['11P']['shiftdata'])
            ax1.plot(data_dic[sample]['ref_ppm_scale'],threeex_fiveex_avg+0.67*WT_offset,'k:',linewidth=lw)
            
            # Population-weighted average using X3 = 23 and X5 = 20
            ex3_ex5_avg = (exch_dic['23']['Q']/exch_dic['WT']['Q'])*np.array(data_dic['23P']['shiftdata']) + (exch_dic['20']['Q']/exch_dic['WT']['Q'])*np.array(data_dic['20P']['shiftdata'])
            ax1.plot(data_dic[sample]['ref_ppm_scale'],ex3_ex5_avg + 0.33*WT_offset,'k-.',linewidth=lw)
            
            # Population-weighted average using four fully trapped mutants 55, 35, 53, 33 = 1120, 1420, 1123, 1423
            pop_avg = exch_dic['WT']['P1120']*np.array(data_dic['1120P']['shiftdata']) + exch_dic['WT']['P1420']*np.array(data_dic['1420P']['shiftdata']) + exch_dic['WT']['P1123']*np.array(data_dic['1123P']['shiftdata']) + exch_dic['WT']['P1423']*np.array(data_dic['1423P']['shiftdata'])
            ax1.plot(data_dic[sample]['ref_ppm_scale'],pop_avg,'k--',linewidth=lw)
            
            ax1.text(0.00, 0.95,'c',horizontalalignment='left', verticalalignment='top',transform=ax1.transAxes,fontsize=30,fontweight='bold')
            #ax1.text(0.96, 0.80,'WT',horizontalalignment='left', verticalalignment='top',transform=ax1.transAxes,fontsize=18,fontweight='bold') # Main text Fig 3
            #ax1.text(0.91, 0.15,'5X+3X',horizontalalignment='left', verticalalignment='top',transform=ax1.transAxes,fontsize=18,fontweight='bold')
            #ax1.text(0.91, 0.37,'X3+X5',horizontalalignment='left', verticalalignment='top',transform=ax1.transAxes,fontsize=18,fontweight='bold')
            #ax1.text(0.82, 0.60,'33+53+35+55',horizontalalignment='left', verticalalignment='top',transform=ax1.transAxes,fontsize=18,fontweight='bold')
            
        if sample == '1423P':
            ax2.plot(data_dic[sample]['ref_ppm_scale'],data_dic[sample]['shiftdata'],color=colors[0],linewidth=lw,label='33')
            ax2.text(0.00, 0.95,'a',horizontalalignment='left', verticalalignment='top',transform=ax2.transAxes,fontsize=30,fontweight='bold')
            for label in ax2.xaxis.get_ticklabels():
                label.set_visible(False)
                
        if sample == '1123P':
            ax3.plot(data_dic[sample]['ref_ppm_scale'],data_dic[sample]['shiftdata'],color=colors[1],linewidth=lw,label='53')
            for label in ax3.xaxis.get_ticklabels():
                label.set_visible(False)
                
        if sample == '1420P':
            ax4.plot(data_dic[sample]['ref_ppm_scale'],data_dic[sample]['shiftdata'],color=colors[2],linewidth=lw,label='35')
            for label in ax4.xaxis.get_ticklabels():
                label.set_visible(False)
                
        if sample == '1120P':
            ax5.plot(data_dic[sample]['ref_ppm_scale'],data_dic[sample]['shiftdata'],color=colors[3],linewidth=lw,label='55')
            
        if sample == '11P':
            ax6.plot(data_dic[sample]['ref_ppm_scale'],data_dic[sample]['shiftdata']+offset,color=colors[4],linewidth=lw,label='5X')
            pop_avg = exch_dic['11']['P1120']*np.array(data_dic['1120P']['shiftdata']) + exch_dic['11']['P1123']*np.array(data_dic['1123P']['shiftdata'])
            ax6.plot(data_dic[sample]['ref_ppm_scale'],pop_avg,'--',color=colors[4],linewidth=lw)
            ax6.text(0.00, 0.95,'b',horizontalalignment='left', verticalalignment='top',transform=ax6.transAxes,fontsize=30,fontweight='bold')
            for label in ax6.xaxis.get_ticklabels():
                label.set_visible(False)
                
        if sample == '14P':
            ax7.plot(data_dic[sample]['ref_ppm_scale'],data_dic[sample]['shiftdata']+offset,color=colors[5],linewidth=lw,label='3X')
            pop_avg = exch_dic['14']['P1420']*np.array(data_dic['1420P']['shiftdata']) + exch_dic['14']['P1423']*np.array(data_dic['1423P']['shiftdata'])
            ax7.plot(data_dic[sample]['ref_ppm_scale'],pop_avg,'--',color=colors[5],linewidth=lw)
            for label in ax7.xaxis.get_ticklabels():
                label.set_visible(False)
                
        if sample == '20P':
            ax8.plot(data_dic[sample]['ref_ppm_scale'],data_dic[sample]['shiftdata']+offset,color=colors[6],linewidth=lw,label='X5')
            pop_avg = exch_dic['20']['P1120']*np.array(data_dic['1120P']['shiftdata']) + exch_dic['20']['P1420']*np.array(data_dic['1420P']['shiftdata'])
            ax8.plot(data_dic[sample]['ref_ppm_scale'],pop_avg,'--',color=colors[6],linewidth=lw)
            for label in ax8.xaxis.get_ticklabels():
                label.set_visible(False)
                
        if sample == '23P':
            ax9.plot(data_dic[sample]['ref_ppm_scale'],data_dic[sample]['shiftdata']+offset,color=colors[7],linewidth=lw,label='X3')
            pop_avg = exch_dic['23']['P1123']*np.array(data_dic['1123P']['shiftdata']) + exch_dic['23']['P1423']*np.array(data_dic['1423P']['shiftdata'])
            ax9.plot(data_dic[sample]['ref_ppm_scale'],pop_avg,'--',color=colors[7],linewidth=lw)
    
    # Set axis properties for each plot
    for idx, ax in enumerate(ax_list):
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.set_yticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(direction='out',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True)
        leg = ax.legend(loc='upper right',frameon=False, handlelength=0)
        for label in ax.xaxis.get_ticklabels()[0::2]:
            label.set_visible(False)
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            line.set_linewidth(3.0)
            text.set_color(line.get_color())
        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_legend().get_texts()):
            item.set_fontsize(18)
            item.set_fontweight('bold')
    
    ax1.set_ylim([-1e6,3e7]) # WT iminos
    #ax1.set_ylim([-1e6,1.5e8]) # WT bases
    #ax1.set_ylim([-1e6,5e7]) # WT sugars
    # Add x-axis label
    fig.subplots_adjust(bottom=0.1)
    fig.text(0.47,0.05,'$^{1}H$ ppm',fontsize=24,fontweight='bold')
    # Save figures
    #plt.savefig('Fig3_Pu22_NMR.png',format='png')
    plt.savefig('Fig3_Pu22_NMR_additionalpopavgs.eps',format='eps')
    #plt.savefig('Fig3_Pu22_NMR_widerview.png',format='png')
    #plt.savefig('Fig3_Pu22_NMR_7point2ppmto8point7ppm_additionalpopavgs.eps',format='eps')
    #plt.savefig('Fig3_Pu22_NMR_full.png',format='png')
    #plt.savefig('Fig3_Pu22_NMR_5point5to6point8ppm_additionalpopavgs.eps',format='eps')
    #plt.show()
    
# Get equilibrium populations for half-trapped and WT sequences for weighting the NMR spectra
def populations():
    
    # Set up sample and population dictionaries
    samples = ['1120','1420','1123','1423']
    ex_samples = ['11','14','20','23','WT']
    Twant = 25 + 273.15
    T0 = 37 + 273.15
    R = 8.3145e-3
    kin_params = {sample:{'kf0':[],'Ef':[],f"kf{str(Twant-273.15)}":[],'ku0':[],'Eu':[],f"ku{str(Twant-273.15)}":[]} for sample in samples}
    thermo_params = {sample:{f"KA{str(Twant-273.15)}":[]} for sample in samples}
    ex_dic = {'11':{'P1120':[],'P1123':[]}, '14':{'P1420':[],'P1423':[]}, '20':{'P1120':[],'P1420':[]}, '23':{'P1123':[],'P1423':[]}, 'WT':{'P1120':[],'P1420':[],'P1123':[],'P1423':[]}}
    
    # Fill in kinetic parameter dictionary based on best global fit TH parameters from Chris
    kin_params['1120']['kf0'] = 0.387
    kin_params['1120']['Ef'] = -36.0
    kin_params['1120']['ku0'] = 0.162
    kin_params['1120']['Eu'] = 121
    kin_params['1420']['kf0'] = 0.572
    kin_params['1420']['Ef'] = -48
    kin_params['1420']['ku0'] = 0.072
    kin_params['1420']['Eu'] = 140
    kin_params['1123']['kf0'] = 0.92
    kin_params['1123']['Ef'] = -55
    kin_params['1123']['ku0'] = 0.023
    kin_params['1123']['Eu'] = 144
    kin_params['1423']['kf0'] = 1.22
    kin_params['1423']['Ef'] = -54.4
    kin_params['1423']['ku0'] = 0.0072
    kin_params['1423']['Eu'] = 164
    
    # Calculate kinetic and thermodynamic parameters at the NMR experiment temperature using Arrhenius relationship
    for sample in samples:
        kf0 = kin_params[sample]['kf0']
        Ef = kin_params[sample]['Ef']
        ku0 = kin_params[sample]['ku0']
        Eu = kin_params[sample]['Eu']
        kin_params[sample][f"kf{str(Twant-273.15)}"] = kf0*np.exp((Ef/R)*((1./(T0)) - (1./(Twant))))
        kin_params[sample][f"ku{str(Twant-273.15)}"] = ku0*np.exp((Eu/R)*((1./(T0)) - (1./(Twant))))
        thermo_params[sample][f"KA{str(Twant-273.15)}"] = kin_params[sample][f"kf{str(Twant-273.15)}"]/kin_params[sample][f"ku{str(Twant-273.15)}"]
    
    # Calculate partition functions for each exchanging construct, either half-trapped or WT
    # Calculate populations of each state in each exchanging structure according to the partition function
    for ex_sample in ex_samples:
        if ex_sample == '11':
            Q = 1. + thermo_params['1120'][f"KA{str(Twant-273.15)}"] + thermo_params['1123'][f"KA{str(Twant-273.15)}"]
            ex_dic[ex_sample]['P1120'] = thermo_params['1120'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['P1123'] = thermo_params['1123'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['Q'] = Q
            
        if ex_sample == '14':
            Q = 1. + thermo_params['1420'][f"KA{str(Twant-273.15)}"] + thermo_params['1423'][f"KA{str(Twant-273.15)}"]
            ex_dic[ex_sample]['P1420'] = thermo_params['1420'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['P1423'] = thermo_params['1423'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['Q'] = Q
            
        if ex_sample == '20':
            Q = 1. + thermo_params['1120'][f"KA{str(Twant-273.15)}"] + thermo_params['1420'][f"KA{str(Twant-273.15)}"]
            ex_dic[ex_sample]['P1120'] = thermo_params['1120'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['P1420'] = thermo_params['1420'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['Q'] = Q
            
        if ex_sample == '23':
            Q = 1. + thermo_params['1123'][f"KA{str(Twant-273.15)}"] + thermo_params['1423'][f"KA{str(Twant-273.15)}"]
            ex_dic[ex_sample]['P1123'] = thermo_params['1123'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['P1423'] = thermo_params['1423'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['Q'] = Q
            
        if ex_sample == 'WT':
            Q = 1. + thermo_params['1120'][f"KA{str(Twant-273.15)}"] + thermo_params['1420'][f"KA{str(Twant-273.15)}"] + thermo_params['1123'][f"KA{str(Twant-273.15)}"] + thermo_params['1423'][f"KA{str(Twant-273.15)}"]
            ex_dic[ex_sample]['P1120'] = thermo_params['1120'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['P1420'] = thermo_params['1420'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['P1123'] = thermo_params['1123'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['P1423'] = thermo_params['1423'][f"KA{str(Twant-273.15)}"]/Q
            ex_dic[ex_sample]['Q'] = Q
            
    print('\n####### Exchanging sequence populations are: #######')
    for k,v in ex_dic.items():
        print(k,v)
    print('\n')
    
    return ex_dic
    
plot_1DNMR()
populations()