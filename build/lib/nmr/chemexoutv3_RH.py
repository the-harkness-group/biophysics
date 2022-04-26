#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:47:22 2019

@author: toyam

191025 including double range

"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.backends.backend_pdf

def main():
    
    # Read in data to plot and convert to ChemEx format
    params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(params['Peakipy fits'])
    groups = data.groupby("assignment")
    
    makeChemEx(params, groups)

def makeChemEx(params, groups):
    
    # Define bigT constant CPMG delay and threshold for delta R2 to use for fits
    bigT = params['Big T']
    threshold = params['Delta R2 threshold']
    
    # Setup lists needed for calculations of R2,eff
    names=[]
    R2all=np.zeros(0)
    R2errorall=np.zeros(0)
    
    # Plot settings
    pdf = matplotlib.backends.backend_pdf.PdfPages("Experimental_CPMG_plots.pdf")
    label_params = {'mathtext.default': 'regular'}
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    
    # Write ChemEx directory structure and headers for files
    os.makedirs(f"../chemex_analysis/Data/{np.round(params['B0'])}") # Uses B0 field to make data directory for ChemEx
    os.makedirs('../chemex_analysis/Experiments')
    os.makedirs('../chemex_analysis/Input')
    os.makedirs('../chemex_analysis/Methods')
    os.makedirs('../chemex_analysis/Output')
    os.makedirs('../chemex_analysis/Parameters')
    
    run_file = open('../chemex_analysis/run.sh','w') # Run file for executing ChemEx
    run_file.write('#!/bin/sh\n\n')
    run_file.write('chemex fit -e Experiments/*.cfg \\\n')
    run_file.write('           -p Parameters/*.cfg \\\n')
    run_file.write('           -m Methods/*.cfg \\\n')
    run_file.write('           -d 2st.pb_kex \\\n')
    run_file.write('           -o Output')
    run_file.close()

    experiment_file = open(f"../chemex_analysis/Experiments/{params['Experiment']}",'w') # ChemEx experiment file header
    experiment_file.write('[experiment]\n')
    experiment_file.write('#name = 600 MHz\n')
    experiment_file.write(f"type = {params['Type']}\n\n")
    experiment_file.write('[experimental_parameters]\n')
    experiment_file.write(f"h1_larmor_frq      = {params['B0']}\n")
    experiment_file.write(f"time_t2            = {params['Big T']}\n")
    experiment_file.write(f"temperature        = {params['Temperature']}\n")
    experiment_file.write('small_protein_flg  = y\n\n')
    experiment_file.write('[extra parameters]\n')
    experiment_file.write('path = ../Data/600\n')
    experiment_file.write('error = duplicates\n\n')
    experiment_file.write('[data]\n')
    
    parameter_file = open(f"../chemex_analysis/Parameters/{params['Parameters']}",'w') # ChemEx parameter file header
    parameter_file.write('[global]\n\n')
    parameter_file.write('#########\n# default fitted parameters\n#########\n\n')
    parameter_file.write('pb                     = 0.3\n')
    parameter_file.write('kex_ab                = 3500.0\n')
    parameter_file.write('dw_ab                 = 0.001\n')
    parameter_file.write('r2_mq_a, B0->600.0MHz = 15.0\n\n')
    parameter_file.write('#########\n# default residue specific\n#########\n\n')
    parameter_file.write('[dw_ab]\n')
    
    method_file = open(f"../chemex_analysis/Methods/{params['Method']}",'w') # Method file for 1H-13C MQ CPMG experiment
    method_file.write('[ step 1 ]\n')
    method_file.write('pb         = fix\n')
    method_file.write('kex_ab     = fix\n')
    method_file.write('dw_ab      = fit\n')
    method_file.write('dw_ab, NUC -> QD1 = fix\n')
    method_file.write('dw_ab, NUC -> QE  = fix\n\n')
    method_file.write('[ step 2 ]\n')
    method_file.write('pb         = fit\n')
    method_file.write('kex_ab     = fit')
    method_file.close()
    
    c13_shifts = open('../chemex_analysis/Input/c13_cs.txt','w') # 13C and 1H chemical shift files
    h1_shifts = open('../chemex_analysis/Input/h1_cs.txt','w')
    
    # Run through list of fitted peaks and calculate experimental CPMG dispersion profiles
    for ind, group in groups:
        vc=np.array(group.vclist)
        amp=np.array(group.amp)
        amp_err=np.array(group.amp_err)
        name=group.assignment.iloc[0]
   
        # convert to R2 and vcpmg
        datapoint=len(vc)
        vcpmg=np.zeros(0)
        R2=np.zeros(0)
        R2err=np.zeros(0)
        maxint=amp[np.where(vc==0)][0]
   
        # Calculate R2,eff for each residue in the supplied list
        for i in range(datapoint):
            if vc[i]>0:
                vcpmg=np.append(vcpmg,vc[i]/bigT)
                R2=np.append(R2,-1./bigT*np.log(amp[i]/maxint))
                R2err=np.append(R2err,1./bigT/amp[i]*amp_err[i])
            else:
                pass
   
        R2b=np.vstack((np.vstack((vcpmg,R2)),R2err))   
        R2b_sort=R2b[:, R2b[0,:].argsort()]
   
        # Plot CPMG dispersions for each residue
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(str(name),fontsize=14)
        ax.set_xlabel("$\\nu_{CPMG}$ Hz",fontsize=14)
        ax.set_ylabel('$R_{2,eff}$ $s^{-1}$',fontsize=14)  
        ax.set_ylim(np.min(R2b_sort[1])-5,np.max(R2b_sort[1]+5))
        ax.errorbar(R2b_sort[0],R2b_sort[1],yerr=R2b_sort[2],xerr=None,fmt='ko',capsize=2)
        ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        ax.yaxis.major.formatter._useMathText = True
        ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        pdf.savefig(fig)
        R2all=np.append(R2all,R2b_sort[1])
        R2errorall=np.append(R2errorall,R2b_sort[2])
        names.append(name)
        plt.close(fig)

        # Filter dispersion profiles based on threshold for exchange rate, write to ChemEx files
        if  R2b_sort[1][0]-R2b_sort[1][len(vcpmg)-1]>threshold:
            R2a=np.vstack((np.vstack((vc,amp)),amp_err))   
            np.savetxt(f"../chemex_analysis/Data/{np.round(params['B0'])}/"+str(name)+".out",np.transpose(R2a))
            
            c13_string = group.assignment.iloc[0].split('_')[2] + 'CD1 ' # Write to 13C shift file
            c13_cs = group.assignment.center_y_ppm.iloc[0]
            c13_ass_shift = c13_string + str(c13_cs) + "\n"
            c13_shifts.write(c13_ass_shift)

            h1_string = group.assignment.iloc[0].split('_')[2] + 'QD1 ' # Write to 1H shift file
            h1_cs = group.assignment.center_x_ppm.iloc[0]
            h1_ass_shift = h1_string + str(h1_cs) + "\n"
            h1_shifts.write(h1_ass_shift)
                        
            experiment_file.write(f"{c13_string} - {h1_string} = {str(name)}.out\n") # Write initial guesses to ChemEx experiment and parameter files
            parameter_file.write(f"{c13_string} = 0.2\n")
            parameter_file.write(f"{1h_string} = 0.001\n")

    pdf.close()
    
    # Close text files that were written
    experiment_file.close()
    parameter_file.close()
    c13_shifts.close()
    h1_shifts.close()

    R2all=np.reshape(R2all,(len(names),int(len(R2all)/len(names))))
    R2all=np.vstack((R2b_sort[0],R2all))

    R2errorall=np.reshape(R2errorall,(len(names),int(len(R2errorall)/len(names))))
    R2errorall=np.vstack((R2b_sort[0],R2errorall))

    np.savetxt('R2all',R2all)
    np.savetxt('R2errorall',R2errorall)

    # Save list of residue names
    f = open('residues.txt', 'w')
    for x in names:
        f.write(str(x) + "\n")
    f.close()

main()