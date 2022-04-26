#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# CESTexperiment
# This script allows you to calculate cest dips from peakipy list.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import minimize, Parameters, fit_report
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages

# this data set includes... even plane Ib, odd plane Ia
try:
    data = pd.read_csv("fits.csv")
except:
    print('DATA FILE MUST BE NAMED fits.csv!')

groups = data.groupby("assignment")

pdf = PdfPages('cestplot.pdf')

for ind,group in groups:
   # data import
   name=group.assignment.iloc[0]
   res=name[1:4]
   print (name)
   
   # Amplitude
   amp = np.array(group.amp)
   # Amp error
   error = np.array(group.amp_err)
   # freq
   freq = np.array(group.vclist)
   freqplot = freq[1:]
      
   # normalize
   # Assuming the first plane is the reference spectrum.
   I0=amp[0]
   error0=error[0]
   
   Ilist = np.empty_like(freqplot)
   Ierrorlist= np.empty_like(freqplot)

   for i in range(len(freqplot)):

       Ilist[i] = amp[i+1]/I0
       Ierrorlist[i]=amp[i+1]/I0*np.sqrt((error[i+1]/I0)**2+(amp[i+1]*error0/I0**2)**2)   
   
   fig1 = plt.figure(figsize=(6,4))
   ax1 = fig1.add_subplot(111)
   ax1.plot(freqplot,Ilist,color='black',ls="-")
   ax1.scatter(freqplot,Ilist,color='black',marker='o',facecolor="None")
   #ax1.errorbar(freqplot,Ilist,fmt='none',yerr=Ierrorlist,ecolor="black")
   ax1.fill_between(freqplot,Ilist-Ierrorlist,Ilist+Ierrorlist,color='black',alpha=0.4)

   ax1.set_ylabel('$I$/$I_{0}$',fontsize=14)
   ax1.set_xlabel('$^{15}N$ frequency [Hz]',fontsize=14)
   ax1.yaxis.major.formatter._useMathText = True
   ax1.set_title(name,fontsize=14)
   #ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
   ax1.tick_params(direction='out',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=12)
   ax1.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
   plt.tight_layout()
   pdf.savefig()
   plt.clf()
   plt.close('all')
pdf.close()

# Write files containing lists of intensity and peak fit error values for each residue, used for ChemEx fitting
for ind, group in groups:
    cest_file = open(f"{ind}.out",'w')
    cest_file.write('#CEST offset (Hz), Peak Intensity, Intensity Error\n')
    for idx,value in enumerate(group.assignment):
        cest_file.write(f"{group['vclist'].iloc[idx]} {group['amp'].iloc[idx]} {group['amp_err'].iloc[idx]}\n")
    cest_file.close()

