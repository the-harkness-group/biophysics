#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:27:48 2020

@author: robertharkness
"""

import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf

# Read-in data and set up dataframe for plotting
def main():
    
    # Read parameters and dataset
    params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(params['Peakipy fits'])
    groups = data.groupby("assignment",sort=False)
    
    # Calculate delta R2s
    delta_R2s = get_deltaR2s(params, groups)
    
    # Plot delta R2 map
    plot_deltaR2map(params, delta_R2s)
    
    #out_df.to_csv(params['Plot_filename'].split('.')[0] + '.csv') # Write dataframe to csv file so you can inspect it


# Calculate delta R2 values from CPMG dispersion profiles for each residue
# Dummy assignments for residues that are not known have an arbitrary number + chemical group, e.g. 5N-H
def get_deltaR2s(params, groups):
    
    # Define bigT constant CPMG delay and threshold for delta R2 to use for fits
    bigT = params['Big T']
    
    # Setup dictionary needed for calculations of R2,eff
    delta_R2s = {}
    
    # Run through list of fitted peaks and calculate R2effs from CPMG data
    for ind, group in groups:
        
        # Get vc, amplitudes, amp fitting error, assignment
        sorted_group = group.sort_values(by=['vclist'],axis=0,ascending=True) # Sort data so that largest R2 is second afer reference experiment (small vcpmg), smallest R2 is last (large vcpmg)
        vc = np.array(sorted_group.vclist)
        amp = np.array(sorted_group.amp)
        amp_err = np.array(sorted_group.amp_err)
        
        if params['CPMG type'] == 'NH':
            assignment = sorted_group.assignment.iloc[0].split('N-H')[0]
            omit = [x.split('N-H')[0] for x in params['Omit']]
            
        if assignment in omit:
            print(f"This residue was skipped since it is either dummy assigned or has bad CPMG profile: {assignment}")
            continue
        
        else:
            delta_R2s[assignment] = {}
   
            # Calculate R2,eff for each residue in the supplied list
            maxint = amp[np.where(vc==0)][0]
            vcpmg = vc/bigT
            R2 = (-1./bigT)*np.log(amp/maxint)
            R2err = 1./bigT/(amp*amp_err)
        
            # Get delta R2 as the difference between the R2eff at vcpmg = min and maximum value
            R2max = R2[1]
            R2min = R2[-1]
            R2maxerr = R2err[1]
            R2minerr = R2err[-1]
            delta_R2 = R2max - R2min
            delta_R2err = R2maxerr - R2minerr
        
            delta_R2s[assignment]['delta_R2'] = delta_R2
            delta_R2s[assignment]['delta_R2err'] = delta_R2err

    return delta_R2s


# Plot fast exchange data as a function of concentration for each peak
def plot_deltaR2map(params, delta_R2s):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(params['Plot_filename']) # Set up pdf for saving figures
    label_params = {'mathtext.default': 'regular' }
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2

    full_seq = [x for x in range(int(params['Range'].split('-')[0]),int(params['Range'].split('-')[1])+1)] # Get full sequence index in case there are gaps in the data
        
    # Make CSP plot as a function of residue number for amides
    if params['CPMG type'] == 'NH':
            
        fig2 = plt.figure(figsize=(20,5))
        ax2 = fig2.add_subplot(111)
        ax2.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        ax2.yaxis.major.formatter._useMathText = True
        ax2.set_xlabel("Residue number",fontsize=16)
        ax2.set_ylabel('$\Delta$ $R_{2}$ [$s^{-1}$]',fontsize=16)
        ax2.set_xlim(min(full_seq)-1,max(full_seq)+1)

        mean = np.array([delta_R2s[k]['delta_R2'] for k in delta_R2s.keys()]).mean()
        std = np.array([delta_R2s[k]['delta_R2'] for k in delta_R2s.keys()]).std()
        
        ax2.axhline(y=mean,color=params['Mean_color'],linestyle='--',linewidth=2) # Plot horizontal lines for mean and mean + 2*std delta R2
        ax2.axhline(y=mean+2.*std,color=params['Std_color'],linestyle='--',linewidth=2)
        ax2.set_xticks(full_seq)
        ax2.set_xticklabels(full_seq)
        ax2.tick_params(top=False,right=False,direction='out',labelsize=14)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
            
        for res in delta_R2s.keys():
            
            if res[0].isdigit(): # Check for dummy assigned residues
                
                print(f"This residue was skipped since it is dummy assigned: {res}")
                continue
            
            else:
                
                try:
                
                    print(f"The current residue is : {res}") # res[1:] is just the residue number, since res[0:] is amino acid + number
                    ax2.bar(int(res[1:]),delta_R2s[res]['delta_R2'],color=params['Bar_color'],edgecolor=params['Bar_color'],width=0.5)
                    
                except:
                    print(f"\n### There is a bad assignment: {res} ###")
                    print("### Check your CPMG plots! ###\n")
            
        for idx,label in enumerate(ax2.xaxis.get_ticklabels()):
            if (idx % 5) != 0:
                label.set_visible(False)
                
        pdf.savefig(fig2)
        plt.close(fig2)
    
    pdf.close()
    
main()