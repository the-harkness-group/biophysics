#!/usr/bin/env python3

import sys
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

### Read in procDLS.yaml file as first argument to get parameters for plotting
with open(sys.argv[1],'r') as stream:
        params = yaml.safe_load(stream)
        
df = pd.read_csv(params['Outname'])

### Updated to run through yaml file containing experiment info for multiple samples
for Sample in params['Samples']:        

    Wells = params['Samples'][Sample]['Wells']
    color_idx = np.linspace(0,1,len(Wells))

### Plot data for a ligand binding DLS experiment
    if params['Samples'][Sample]['Experiment type'] == 'Binding':
    
        ### Start pdf file for saving figures
        pdf = matplotlib.backends.backend_pdf.PdfPages(params['Samples'][Sample]['Plotting filename'])
    
        ### Calculate protein concentration in samples
        Protein_concentration = (params['Samples'][Sample]['Protein volume']/params['Total sample volume'])*params['Samples'][Sample]['Protein stock concentration']
        Protein_concentration = round(Protein_concentration,1)
    
        ### Calculate ligand concentration in samples
        vol_ligand = np.array(params['Samples'][Sample]['Ligand volume'])
        conc_ligand = (np.array(params['Samples'][Sample]['Ligand volume'])/params['Total sample volume'])*params['Samples'][Sample]['Ligand stock concentration']
    
        ### Plot first figure with D versus T
        fig_1 = plt.figure()
        ax_1 = fig_1.add_subplot(111)
        ax_1.set_xlabel('Temperature $^\circ$C')
        ax_1.set_ylabel('$D_{t}$ cm\u00b2 s\u207b\u00b9')
        ax_1.set_title(f"[{Sample}] = {Protein_concentration} $\mu$M")
        ax_1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        ax_1.yaxis.major.formatter._useMathText = True
        conc_index = 0
        for x,well in zip(color_idx,Wells):
            well_df = df[df['Well']==well]
            ax_1.plot(well_df['Temperature'],well_df['D']*1e4,color=plt.cm.cool(x),marker='o',linestyle='None',label=f"{str(round(conc_ligand[conc_index],1))}")
            conc_index = conc_index + 1
        
        legend = ax_1.legend(title=f"[{params['Samples'][Sample]['Ligand name']}] $\mu$M")
        pdf.savefig(fig_1)
              
    ### Plot additional figures with D versus ligand concentration at various temperatures
        for Temperature in params['Temperature']:
            try:
                fig_2 = plt.figure()
                ax_2 = fig_2.add_subplot(111)
                ax_2.set_xlabel(f"[{params['Samples'][Sample]['Ligand name']}] $\mu$M")
                ax_2.set_ylabel('$D_{t}$ $cm^{2}$ $s^{-1}$')
                ax_2.set_title(f"[{Sample}] = {Protein_concentration} $\mu$M, T = {Temperature} $^\circ$C")
                ax_2.ticklabel_format(style='sci',axis='y')
                ax_2.yaxis.major.formatter._useMathText = True
                conc_index = 0
                for x,well in zip(color_idx,Wells):
                    well_df = df[df['Well'] == well]
                    well_df = well_df[round(well_df['Temperature'])==Temperature]
                    ax_2.plot(conc_ligand[conc_index],well_df['D']*1e4,color=plt.cm.cool(x),marker='o',linestyle='None')
                    conc_index = conc_index+1
        
                pdf.savefig(fig_2)
                
            except:
                print(f"DLS data for {Temperature} degrees are missing")
            pdf.close()
            plt.close('all')

### Plot data for a concentration series/dilution-type DLS experiment
    elif (params['Samples'][Sample]['Experiment type'] == 'Dilution') or (params['Samples'][Sample]['Experiment type'] == 'Sweep'):

        ### Calculate protein concentrations in the dilution series
        if params['Samples'][Sample]['Experiment type'] == 'Dilution':
            Protein_concentration = np.array([params['Samples'][Sample]['Protein stock concentration']*params['Samples'][Sample]['Dilution factor']**x for x in range(len(Wells))])
            print(f"The {Sample} concentrations are: {Protein_concentration} \u03BCM")
    
        ### Calculate protein concentrations in concentration sweep
        elif params['Samples'][Sample]['Experiment type'] == 'Sweep':
            Protein_concentration = np.array(params['Samples'][Sample]['Protein concentration'])
    
        color_idx = np.linspace(0,1,len(Wells))
        fig_Ts = params['Temperature']
        fig_handles = [plt.figure() for x in range(len(fig_Ts))]
        fig_axes = [fig_handles[x].add_subplot(111) for x in range(len(fig_handles))]

        for idx,Temperature in zip(range(len(fig_Ts)),fig_Ts):
            conc_index = 0
            for x,well in zip(color_idx,Wells):
                try:
                    subdf = df[df['Well']==well]
                    Tdf= subdf[round(subdf['Temperature'])==round(float(Temperature))]                    
                    fig_axes[idx].plot(Protein_concentration[conc_index],Tdf['D']*1e4,color=plt.cm.cool(x),marker='o',linestyle='None')
                    conc_index = conc_index+1
                    
                    fig_axes[idx].set_xlabel(f"[{Sample}] $\mu$M")
                    fig_axes[idx].set_ylabel('$D_{z}$  $cm^{2}$  $s^{-1}$')
                    fig_axes[idx].set_title('Temperature = {} $^\circ$C'.format(fig_Ts[idx]))    
                    fig_axes[idx].yaxis.major.formatter._useMathText = True
                    fig_axes[idx].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
                    
                except:
                    print(f"***D value is missing for {Sample} well {well} at {Temperature} degrees***")
                    fig_axes[idx].set_xlabel(f"[{Sample}] $\mu$M")
                    fig_axes[idx].set_ylabel('$D_{z}$  $cm^{2}$  $s^{-1}$')
                    fig_axes[idx].set_title('Temperature = {} $^\circ$C ###NO DATA AVAILABLE###'.format(fig_Ts[idx]))    
                    fig_axes[idx].yaxis.major.formatter._useMathText = True
                    fig_axes[idx].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
                    
                    conc_index = conc_index+1

        D_fig = plt.figure()
        D_axis = D_fig.add_subplot(111)
        D_axis.set_xlabel('Temperature $^\circ$C')
        D_axis.set_ylabel('$D_{z}$  $cm^{2}$  $s^{-1}$')
        D_axis.set_title(f"[{Sample}] stock = {params['Samples'][Sample]['Protein stock concentration']} $\mu$M")
        D_axis.yaxis.major.formatter._useMathText = True
        D_axis.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        conc_index = 0
        for x,well in zip(color_idx,Wells):
            Protein_df = df[df['Well']==well]
            D_axis.plot(Protein_df['Temperature'],Protein_df['D']*1e4,color=plt.cm.cool(x),marker='o',linestyle='None',label=f"{str(round(Protein_concentration[conc_index],1))}")
            conc_index = conc_index + 1
            
        legend = D_axis.legend(title=f"[{Sample}] $\mu$M")
        
        pdf = matplotlib.backends.backend_pdf.PdfPages(params['Samples'][Sample]['Plotting filename'])
        pdf.savefig(D_fig)
        for figure in fig_handles:
            pdf.savefig(figure)
        pdf.close()
        plt.show()
        plt.close('all')    