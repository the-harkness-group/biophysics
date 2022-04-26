#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:58:49 2020

@author: robertharkness
"""

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
    groups = data.groupby("assignment",sort=False)
    
    plotCPMG(params, groups)

def plotCPMG(params, groups):
    
    # Define bigT constant CPMG delay
    bigT = params['Big T']
    
    # Plot settings
    pdf = matplotlib.backends.backend_pdf.PdfPages(params['PDF_name'])
    label_params = {'mathtext.default': 'regular'}
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    
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
        ax.errorbar(R2b_sort[0],R2b_sort[1],yerr=R2b_sort[2],xerr=None,fmt='ko',capsize=2)
        ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        ax.yaxis.major.formatter._useMathText = True
        ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()

main()