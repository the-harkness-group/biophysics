#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:22:07 2020

@author: robertharkness
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd

def main():
    
    # Read in data to plot and convert to ChemEx format
    params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(params['Peakipy fits'])
    groups = data.groupby("assignment",sort=False)
    dispersions = params['Dispersions']
    
    make_chemex(params, groups, dispersions)

def make_chemex(params, groups, dispersions):
    
    # Define bigT constant CPMG delay and threshold for delta R2 to use for fits
    bigT = params['Big T']
    
    # Write ChemEx directory structure and headers for files
    os.makedirs(f"../{params['CPMG_experiment']}/Data/{str(params['B0']).split('.')[0] + 'MHz'}") # Uses B0 field to make data directory for ChemEx
    os.makedirs(f"../{params['CPMG_experiment']}/Experiments")
    os.makedirs(f"../{params['CPMG_experiment']}/Methods")
    os.makedirs(f"../{params['CPMG_experiment']}/Parameters")
    
    run_file = open(f"../{params['CPMG_experiment']}/run.sh",'w') # Run file for executing ChemEx
    run_file.write('#!/bin/sh\n\n')
    run_file.write('chemex fit -e Experiments/*.toml \\\n')
    run_file.write('           -p Parameters/*.toml \\\n')
    run_file.write('           -m Methods/*.toml \\\n')
    run_file.write('           -o Output')
    run_file.close()
    
    parameter_file = open(f"../{params['CPMG_experiment']}/Parameters/{params['Parameters']}",'w') # ChemEx parameter file header
    parameter_file.write('[CS_A]\n')
    
    # Run through list of fitted peaks and calculate experimental CPMG dispersion profiles
    for ind, group in groups:
        vc=np.array(group.vclist)
        amp=np.array(group.amp)
        amp_err=np.array(group.amp_err)
   
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

        # Filter dispersion profiles based on threshold for exchange rate, write to ChemEx files
        if  ind in dispersions:
            R2a=np.vstack((np.vstack((vc,amp)),amp_err))
            np.savetxt(f"../{params['CPMG_experiment']}/Data/{str(params['B0']).split('.')[0] + 'MHz'}/"+str(ind)+".out",np.transpose(R2a))
            if params['CPMG_type'] == 'NH':
                parameter_file.write(f"{ind.split('N')[0] + 'HN'} = {group.center_x_ppm.iloc[0]}\n")
            
            elif params['CPMG_type'] == 'N15':
                parameter_file.write(f"{ind.split('N')[0] + 'N'} = {group.center_y_ppm.iloc[0]}\n")
    
    # Close text files that were written
    parameter_file.close()


main()