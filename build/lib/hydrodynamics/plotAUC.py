#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:25:54 2021

@author: robertharkness
"""

import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import customcolormap
import matplotlib.backends.backend_pdf

def plotAUC():
    
    params = yaml.safe_load(open(sys.argv[1],'r'))
    files = params['Files']
    
    hex_colors = ['#020101','#281112','#4E2128','#733243','#984364','#BD548A','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF']
    hex_colors = list(reversed(hex_colors))
    temperature_colors = ['#4575b4','#91bfdb','#66c2a4']
    
    plt.style.use('figure')
    pdf = matplotlib.backends.backend_pdf.PdfPages('DegP2_AUC_plots.pdf')
    
    for file in files:
        
        data = pd.read_csv(file)
        data.columns = ['s20w','cs']
        temperature = file.split('_')[2].split('C')[0] # DegP_high/lowsalt_temperatureC.csv'
        
        startindex = 0
        colorindex = 0
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('$s_{20,w}$ S')
        ax.set_ylabel('Normalized c(s)')
        ax.set_ylim([0,5])
        ax.set_xlim([3,15])
        #ax.text(0.7,0.85,'S210A',fontsize=24,transform=ax.transAxes,va="top")
        #ax.text(0.7,0.85,'S210A',fontsize=24,transform=ax.transAxes,va="top")
        
        if temperature == '8':
            #ax.set_title(f"{temperature} \N{DEGREE SIGN}C",color=temperature_colors[0])
            
            if 'highsalt' in file:
                
                trimerdata = pd.read_csv(params['Trimer Files'][params['Trimer Files'].index('DegP_S210AY444A_highsalt_8C.csv')])
                trimerdata.columns = ['s20w','cs']
                ax.plot(trimerdata['s20w'][:-1],trimerdata['cs'][:-1],linewidth=2,color='#fc8d59')
                ax.text(0.2,0.6,'$M_{3}$',color='#fc8d59',fontsize=24,transform=ax.transAxes,va="top")
            
        if temperature == '20':
            #ax.set_title(f"{temperature} \N{DEGREE SIGN}C",color=temperature_colors[1])
            
            if 'highsalt' in file:
                
                trimerdata = pd.read_csv(params['Trimer Files'][params['Trimer Files'].index('DegP_S210AY444A_highsalt_20C.csv')])
                trimerdata.columns = ['s20w','cs']
                ax.plot(trimerdata['s20w'][:-1],trimerdata['cs'][:-1],linewidth=2,color='#fc8d59')
                ax.text(0.2,0.45,'$M_{3}$',color='#fc8d59',fontsize=24,transform=ax.transAxes,va="top")
        
        if temperature == '30':
            #ax.set_title(f"{temperature} \N{DEGREE SIGN}C",color=temperature_colors[2])
            
            if 'highsalt' in file:
                
                trimerdata = pd.read_csv(params['Trimer Files'][params['Trimer Files'].index('DegP_S210AY444A_highsalt_30C.csv')])
                trimerdata.columns = ['s20w','cs']
                ax.plot(trimerdata['s20w'][:-1],trimerdata['cs'][:-1],linewidth=2,color='#fc8d59')
                ax.text(0.2,0.3,'$M_{3}$',color='#fc8d59',fontsize=24,transform=ax.transAxes,va="top")
        
        
        separator = data[data['s20w'] < -100]
        cmap = customcolormap.get_continuous_cmap(hex_colors,len(separator))
        
        for index, value in enumerate(data['s20w']):
            
            if data['s20w'][index] < -100:
                
                endindex = index-1 # i-1 from -999.999000 ie the last value of the data series
                
                ax.plot(data['s20w'][startindex:endindex],data['cs'][startindex:endindex],linewidth=2,color=cmap[colorindex])
                
                startindex = index+1 # Update plot start index to be i+1 from -0.999, ie start of next data series
                colorindex +=1 # Update color index to plot next concentration in new color
                
                #plt.show()
        
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(f"{file.split('csv')[0][:-1]}.png",format='png')
        plt.close(fig)
    
    pdf.close()

plotAUC()