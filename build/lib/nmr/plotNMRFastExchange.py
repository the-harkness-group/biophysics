#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:53:46 2020

@author: robertharkness
"""

import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

# Read-in data and set up dataframe for plotting
def main():
    
    params = yaml.safe_load(open(sys.argv[1],'r'))
    NMR_peaklists = params['Lists']
    Concentrations = params['Concentrations']
    
    out_df, CSPs = parse_data(params, NMR_peaklists, Concentrations)
    plot_titration(params, out_df, CSPs)
    
    out_df.to_csv(params['Plot_filename'].split('.')[0] + '.csv') # Write dataframe to csv file so you can inspect it


# Make fast exchange dataframe from NMR peak lists and experimental concentrations
# Can handle dilution or mixing type experiments
def parse_data(params, NMR_peaklists, Concentrations):
    
    # Make initial dataframe which has all of the data read into single table format
    df_list = []
    for NMR_list, Concentration in zip(NMR_peaklists, Concentrations):
        
        data = pd.read_csv(NMR_list,sep='\s+')
        data.columns = ['Assignment','w1_ppm','w2_ppm','w1_Hz','w2_Hz','blank','blank'] # Peak list has to have this format from Sparky
        data = data[['Assignment','w1_ppm','w2_ppm','w1_Hz','w2_Hz']]
        df_conc = [Concentration for x in range(len(data['Assignment']))]
        df_experiment = [NMR_list for x in range(len(data['Assignment']))]
        
        data.insert(3,'CSP_ppm',0,True)
        data.insert(6,'CSP_Hz',0,True)
        data.insert(7,'Concentration_uM',df_conc,True)
        data.insert(8,'Experiment_peaklist',df_experiment,True)
        
        df_list.append(data)
    fastex_df = pd.concat(df_list)
    
    # Write another dataframe which has the chemical shift perturbations calculated relative to the lowest concentration
    groups = fastex_df.groupby('Assignment',sort=False)
    df_list = []
    CSPs = {}
    for ind, group in groups:
        
        CSPs[ind] = {'ppm':[],'Hz':[]}
        
        group = group.sort_values('Concentration_uM',ascending=False) # Put into descending concentration order so that this works for both dilution and mixing type experiments
        
        if params['Type'] == 'Amides':
            
            group.loc[:,'CSP_ppm'] = np.abs(np.sqrt( np.square( ((group.w1_ppm - group.w1_ppm.iloc[-1])/params['N15_scaling'])) + np.square(group.w2_ppm - group.w2_ppm.iloc[-1]) )) 
            group.loc[:,'CSP_Hz'] = np.abs(np.sqrt( np.square(group.w1_Hz - group.w1_Hz.iloc[-1]) + np.square(group.w2_Hz - group.w2_Hz.iloc[-1]) ))
            CSPs[ind]['ppm'] = group.CSP_ppm.iloc[0] - group.CSP_ppm.iloc[-1]
            CSPs[ind]['Hz'] = group.CSP_Hz.iloc[0] - group.CSP_Hz.iloc[-1]
        
        if params['Type'] == 'Methyls':
            if ind[0] == 'T':
                H1_scaling = params['Thr_scaling']['1H']
                C13_scaling = params['Thr_scaling']['13C']
            
            if ind[0] == 'A':
                H1_scaling = params['Ala_scaling']['1H']
                C13_scaling = params['Ala_scaling']['13C']
                
            if ind[0] == 'M':
                H1_scaling = params['Met_scaling']['1H']
                C13_scaling = params['Met_scaling']['13C']
            
            if ind[0] == 'L':
                H1_scaling = params['Leu_scaling']['1H']
                C13_scaling = params['Leu_scaling']['13C']
                
            if ind[0] == 'V':
                H1_scaling = params['Val_scaling']['1H']
                C13_scaling = params['Val_scaling']['13C']
            
            if ind[0] == 'I':
                H1_scaling = params['Ile_scaling']['1H']
                C13_scaling = params['Ile_scaling']['13C']
            
            group.loc[:,'CSP_ppm'] = np.sqrt( np.square( ((group.w1_ppm - group.w1_ppm.iloc[-1])/C13_scaling) ) + np.square( ((group.w2_ppm - group.w2_ppm.iloc[-1])/H1_scaling) ) )    
            group.loc[:,'CSP_Hz'] = np.sqrt( np.square(group.w1_Hz - group.w1_Hz.iloc[-1]) + np.square(group.w2_Hz - group.w2_Hz.iloc[-1]) )
            CSPs[ind]['ppm'] = group.CSP_ppm.iloc[0] - group.CSP_ppm.iloc[-1]
            CSPs[ind]['Hz'] = group.CSP_Hz.iloc[0] - group.CSP_Hz.iloc[-1]
            
        df_list.append(group)
    out_df = pd.concat(df_list)
        
    return out_df, CSPs


# Plot fast exchange data as a function of concentration for each peak
def plot_titration(params, out_df, CSPs):
    
    groups = out_df.groupby('Assignment',sort=False) # Don't sort the groups to preserve sequence order otherwise the bar plot will not be right
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(params['Plot_filename']) # Set up pdf for saving figures
    label_params = {'mathtext.default': 'regular' }
    plt.style.use('figure')
    plt.rcParams.update(label_params)

    for ind, group in groups:
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        ax1.yaxis.major.formatter._useMathText = True
        ax1.set_title(f"{ind}")

        # Plot residue-specific CSPs to inspect for fitting
        if params['Unit'] == 'Hz':
            ax1.plot(group.Concentration_uM,group.CSP_Hz,'ko')
            ax1.set_ylabel('CSP [Hz]')
            
        if params['Unit'] == 'ppm':
            ax1.plot(group.Concentration_uM,group.CSP_ppm,'ko')
            ax1.set_ylabel('CPS [ppm]')
        
        if params['Experiment'] == 'mixing':
            ax1.set_xlabel("$L_{T}$ [$\mu$M]")
    
        if params['Experiment'] == 'dilution':
            ax1.set_xlabel("$M_{T}$ [$\mu$M]")

        fig1.tight_layout()
        #fig1.subplots_adjust(bottom=0.15)
        pdf.savefig(fig1)
        plt.close(fig1)
    
    if params['CSP_map'] == 'y':
        
        full_seq = [x for x in range(int(params['Range'].split('-')[0]),int(params['Range'].split('-')[1])+1)] # Get full sequence index for axis limits
        
        fig2 = plt.figure(figsize=(16,5)) # Set up CSP plot
        ax2 = fig2.add_subplot(111)
        ax2.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        ax2.yaxis.major.formatter._useMathText = True
        ax2.set_xlabel("Residue number")
        if params['Unit'] == 'Hz':
            ax2.set_ylabel('CSP Hz')
        if params['Unit'] == 'ppm':
            ax2.set_ylabel('Scaled $\Delta \delta$ ppm')
        ax2.set_xlim(min(full_seq)-1,max(full_seq)+1)
        
        # Make CSP plot as a function of residue number for amides
        good_CSP_list = [] # For getting mean and standard deviation of CSPs
        if params['Type'] == 'Amides':
            
            ax2.set_title(f"{params['Type']} {params['Temperature']} \N{DEGREE SIGN}C",color=params['Bar color'])
            
            for k in CSPs.keys():
                
                if k in params['Broadened residues']: # Residues that were severely exchange broadened during the titration and couldn't be accurately tracked
                    ax2.plot(int(k.split('N-H')[0][1:]),10,'k*',markersize=10)
                
                if k in params['Overlapped residues']: # Residues that became overlapped during the titration and couldn't be accurately tracked
                    ax2.plot(int(k.split('N-H')[0][1:]),10,'k*',markersize=10)
                
                if k not in params['Broadened residues'] and k not in params['Overlapped residues']: # Residues that were accurately tracked
                    ax2.bar(int(k.split('N-H')[0][1:]),CSPs[k]['Hz'],color=params['Bar color'],edgecolor=params['Bar color'],width=0.5)
                    good_CSP_list.append(CSPs[k]['Hz'])
                    
            for value in params['Missing residues']: # Residues that were broadened out and missing in the assignment process and so couldn't be tracked at all
                ax2.plot(int(value.split('N-H')[0][1:]),10,'k*',markersize=10)
                  
        # Make CSP plot as a function of residue number for methyls
        if params['Type'] == 'Methyls':
            
            ax2.set_title(f"{params['Type']} {params['Temperature']} \N{DEGREE SIGN}C",color=params['Bar color'])
            
            shift = 0.25 # for offsetting residues CSP plot that have two methyls like I, L, V
                        
            for k in CSPs.keys():
                
                if k.split('C')[0][-1] == '?': # Check for residues with assignments that are tentative
                    number = int(k.split('C')[0][1:-1]) # Get methyl residue number for tentatively assigned residues
                    
                else:
                    number = int(k.split('C')[0][1:]) # Get methyl residue number for assigned residues
                
                if k[0] == 'L' or k[0] == 'V' or k[0] == 'I':
                    
                    if k.split('-')[1][-1] == '1': # Offset residue index by +/-0.5 to account for the two methyls
                        number = number - shift
                    
                    if k.split('-')[1][-1] == '2':
                        number = number + shift
                        
                    if k.split('-')[1] == 'HG':
                        number = number - shift
                        
                    if k.split('-')[1] == 'HD':
                        number = number + shift
                
                if k in params['Broadened residues']: # Residues that were severely exchange broadened during the titration and couldn't be accurately tracked
                    ax2.plot(number,10,'k*',markersize=10)
                
                if k in params['Overlapped residues']: # Residues that became overlapped during the titration and couldn't be accurately tracked
                    ax2.plot(number,10,'k*',markersize=10)
                
                if k not in params['Broadened residues'] and k not in params['Overlapped residues'] and k not in params['Missing residues']: # Residues that were accurately tracked
                    ax2.bar(number,CSPs[k]['Hz'],color=params['Bar color'],edgecolor='w',hatch='///',width=0.5)
                    good_CSP_list.append(CSPs[k]['Hz'])
                    
            for value in params['Missing residues']: # Residues that were broadened out and missing in the assignment process and so couldn't be tracked at all
                
                if value.split('C')[0][-1] == '?': # Check for residues with assignments that are tentative
                    number = int(value.split('C')[0][1:-1]) # Get methyl residue number for tentatively assigned residues
                    
                else:
                    number = int(value.split('C')[0][1:]) # Get methyl residue number for assigned residues
                
                if value[0] == 'L' or value[0] == 'V' or value[0] == 'I':
                    
                    if value.split('-')[1][-1] == '1': # Offset residue index by +/-0.5 to account for the two methyls
                        number = number - shift
                    
                    if value.split('-')[1][-1] == '2':
                        number = number + shift
                        
                    if value.split('-')[1] == 'HG':
                        number = number - shift
                        
                    if value.split('-')[1] == 'HD':
                        number = number + shift
                
                ax2.plot(number,10,'k*',markersize=10)
                            
        ax2.axhline(y=np.mean(good_CSP_list),color=params['Mean color'],linestyle='--',linewidth=2) # Plot mean and standard deviation of CSPs from good CSP list
        ax2.axhline(y=np.mean(good_CSP_list)+np.std(good_CSP_list),color=params['Standard deviation color'],linestyle='--',linewidth=2)
              
        # Set tick labels on CSP plot
        ax2.set_xticks(full_seq)
        ax2.set_xticklabels(full_seq)
        ax2.tick_params(top=False,right=False,direction='out')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
            
        for idx,label in enumerate(ax2.xaxis.get_ticklabels()):
            if (idx % 5) != 0:
                label.set_visible(False)
        
        fig2.tight_layout()
        #fig2.subplots_adjust(bottom=0.15)
        pdf.savefig(fig2)
        fig2.savefig('CSP_map.png',format='png')
        fig2.savefig('CSP_map.pdf',format='pdf')
        plt.close(fig2)
    
    pdf.close()
    
main()