#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:06:16 2020

@author: toyam
"""

import sys
import numpy as np
import scipy as sp
from scipy import stats
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import nmrglue as ng
from lmfit import minimize, Parameters, fit_report
import pandas as pd
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

# Read/input data set 
filename = sys.argv[1]
dic,data=ng.pipe.read_3D(filename)

# Spectrum output
contour_start = 1.5E5          # contour level start value
contour_num = 32                # number of contour levels
contour_factor = 1.20          # scaling factor between contour levels
cl = contour_start*contour_factor**np.arange(contour_num)

# Concentration list
#conclist=np.array([0,36.4,72.1,105.8,145.8,205,259.7,314.3,373.5,447.2,519])
#receptorlist=np.repeat(346,11)
conclist=np.repeat(519,11)
receptorlist=np.repeat(346,11)
scalelist=np.ones(len(conclist))
# inept transfer time
tinept=0.0072

# Here we assume each plane was recorded in a pseudo-3d manner.
# The current data has index0: titration point, index1: 13Cy plane=Y, index2: 1H plane=X
udicg = ng.pipe.guess_udic(dic,data)
print (udicg)

# Axis1 = 13C
udic_1=ng.pipe.make_uc(dic,data,dim=1)
ax1_full_array=udic_1.ppm_scale()
# Axis2 = 1H
udic_2=ng.pipe.make_uc(dic,data,dim=2)
ax2_full_array=udic_2.ppm_scale()

# Define an ellipse function to get better boxsum
# a: F1 radius, b: F2 radius
def ellipse(array1,array2,center1,center2,a,b):
  td1=(len(array1))
  td2=(len(array2))
  grid = np.zeros([td1,td2])
  for k in range(td1):
    for l in range(td2):
      if ((array1[k]-center1)/a)**2+((array2[l]-center2)/b)**2<1:
        grid[k,l]=1
  return grid

# data import
peaklist = pd.read_csv("peaklist.csv")

# Output
col1 = ['Assignment', 'State','Concentration','Receptor','Intensity',
        'Corr_Intensity','Height','Max_int','Plane','Index']

out_df = pd.DataFrame(columns=col1)
pdf = PdfPages('peak_integrations.pdf')

for i in peaklist.index:

    for p in range(len(data)):

      if peaklist.at[i,'fit']=='y': 
        
        # data import
        residue=peaklist.at[i,'assignment']
        state=peaklist.at[i,'state']
                
        Hx=peaklist.at[i,'x']
        xidx=udic_2(str(Hx)+'ppm')
        Cy=peaklist.at[i,'y']
        yidx=udic_1(str(Cy)+'ppm')
        
        a=peaklist.at[i,'a']
        b=peaklist.at[i,'b']
        
        grid=ellipse(ax1_full_array,ax2_full_array,Cy,Hx,a,b)
        region=data[p]*grid*scalelist[p]
        volume=np.sum(region)
        height=data[p,yidx,xidx]
        maxint=np.max(region)

        try:
          R=peaklist.at[i,'R2']
          corr_volume=volume*np.exp(R*tinept)
        except:
          corr_volume = volume

        tmp_se = pd.Series([residue, state, conclist[p], receptorlist[p], volume,
                            corr_volume,height,maxint,p,i],
                        index=out_df.columns )
        out_df=out_df.append(tmp_se, ignore_index=True )
               
        # Output the spectrum to check the region is OK
        figsize=(5,5)   
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        X,Y = np.meshgrid(ax2_full_array, ax1_full_array)
        ax.contour(X,Y,data[p]*scalelist[p],cl,colors="Black",linewidths=0.5)
        ax.contour(X,Y,-1*data[p]*scalelist[p],cl,colors="Red",linewidths=0.5)
        ax.contourf(X,Y,grid,alpha=0.25,levels=1,colors=["white","violet"])
        ax.scatter(Hx,Cy, c="hotpink",marker='x')

        ax.set_xlabel("$^{1}$H (ppm)", size=12)
        ax.set_xlim(Hx+0.2,Hx-0.2)
        ax.set_ylim(Cy+0.5,Cy-0.5)
        #ax.invert_xaxis()
        #ax.invert_yaxis()
        ax.set_ylabel("$^{13}$C (ppm)", size=12)
        ax.spines['top'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        plt.tick_params(labelsize=12)
        ax.set_title(residue+", plane "+str(p)+", "+str(state)
        ,horizontalalignment='center', verticalalignment='top', fontsize=16)
        ax.text(Hx,Cy-0.15,"Volume=%.2e"%(volume),fontsize=14,c='violet')
        plt.tight_layout()
        pdf.savefig()
        plt.close("all")
      
      else:
        pass

pdf.close()

try:
    pdf = PdfPages('titration_plots.pdf')
    groups = out_df.groupby('Assignment')
    for ind,group in groups:
      fig,ax = plt.subplots(1,1)
      ax.plot(conclist,group.Intensity,'ko')
      ax.set_xlabel('Concentration uM')
      ax.set_ylabel('Peak volume')
      ax.set_title(ind)
      pdf.savefig()
    pdf.close()
except Exception as e:
    print(e)

out_df = out_df.sort_values(by=['Assignment','Concentration'])
out_df.to_csv("out.csv")

"""
# Add the population line
out_df2 = out_df
out_df2['population']=0.

for i in out_df2.index:
    
    corr_vol=out_df2.at[i,'corr_volume']
    residue=out_df2.at[i,'assignment']
    plane=out_df2.at[i,'plane']
  
    totalvol=np.sum(out_df2[(out_df2['assignment']==residue)
                 & (out_df2['plane']==plane)].corr_volume)
 

    out_df2.at[i,'population']=corr_vol/totalvol

out_df2.to_csv("out.csv")
""" 