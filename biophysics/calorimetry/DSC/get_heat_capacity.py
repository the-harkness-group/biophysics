#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:06:32 2019

@author: robertharkness
"""

import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pickle

# Read in yaml file containing the experiment information
params = yaml.safe_load(open(sys.argv[1],'r'))

# TA NanoDSC with autosampler instrumental constants
# Cell volume needs to be in L
Vcell = 0.3/1000

# Plotting settings
label_params = {'mathtext.default': 'regular' }          
plt.rcParams.update(label_params)
plt.rcParams['axes.linewidth'] = 2

def WaterProperties(Temperature):
    # Calculate heat capacity of water as a function of temperature in Celsius
    # Taken from empirically fit 5th order polynomial to measured water Cp data
    
    Cp_water = np.array([-1e-11*np.power(Temperature[x],5) + 6e-9*np.power(Temperature[x],4) - 1e-6*np.power(Temperature[x],3) + 8e-5*np.square(Temperature[x]) - 0.003*Temperature[x] + 4.2196 for x in range(len(Temperature))])

    # Calculate density of water as a function of temperature in Celsius
    # Taken from empirically fit 4th order polynomial to measured water density data
    p_water = np.array([-7e-11*np.power(Temperature[x],4) + 3e-8*np.power(Temperature[x],3) - 7e-6*np.square(Temperature[x]) + 4e-5*Temperature[x] + 0.9999 for x in range(len(Temperature))])

    return Cp_water, p_water

# Calculate heat capacity data using raw sample and buffer DSC data
pdf = matplotlib.backends.backend_pdf.PdfPages('Processed_DSC_data.pdf')
for sample in params['Samples'].keys():
    # Read experimental sample and buffer DSC data using file names from yaml
    sample_data = pd.read_csv(params['Samples'][sample]['File'])
    buffer_data = pd.read_csv(params['Samples'][sample]['Buffer'])
    
    # Calculate average buffer baseline to subtract from the sample data
    # First make dictionary of buffer data, then conver to numpy array to get the mean buffer scan for subtraction
    buffer_power = {}
    buffer_power = {f"Scan_{str(scan)}":{'Temperature':[],'Power':[]} for scan in params['Samples'][sample]['Buffer scans']}
    fig, axs = plt.subplots(1,2,figsize=(15,4))
    for x,col in enumerate(buffer_data.columns):
        if isinstance(buffer_data[col][0],str):
            for scan in params['Samples'][sample]['Buffer scans']:
                if int(buffer_data[col][0].split(' / ')[1].split(' ')[1]) == scan:
                    
                    buffer_power[f"Scan_{str(scan)}"]['Temperature'] = np.array(buffer_data[col][2:].astype(float))
                    buffer_power[f"Scan_{str(scan)}"]['Power'] = np.array(buffer_data[f"Unnamed: {x+1}"][2:].astype(float))*1e-6
                    
                    # Plot buffer scans
                    axs[0].plot(buffer_power[f"Scan_{str(scan)}"]['Temperature'][100:],buffer_power[f"Scan_{str(scan)}"]['Power'][100:],LineWidth=2,label=f"Buffer scan {str(scan)}")

    mean_buffer_temperature = np.mean([buffer_power[f"Scan_{str(scan)}"]['Temperature'] for scan in params['Samples'][sample]['Buffer scans']],0)
    mean_buffer_power = np.mean([buffer_power[f"Scan_{str(scan)}"]['Power'] for scan in params['Samples'][sample]['Buffer scans']],0)
    # Plot average buffer scan
    axs[0].plot(mean_buffer_temperature[100:],mean_buffer_power[100:],LineWidth=2,label="Average buffer scan")

    # Repeat for sample data except need to subtract the average buffer baseline to get sample power
    sample_power = {}
    sample_power = {f"Scan_{str(scan)}":{'Temperature':[],'Raw_power':[],'Buffer_subtracted_power':[]} for scan in params['Samples'][sample]['Sample scans']}
    sample_Cp = {}
    sample_Cp = {f"Scan_{str(scan)}":{'Temperature':[],'Power':[]} for scan in params['Samples'][sample]['Sample scans']}
    for x,col in enumerate(sample_data.columns):
        if isinstance(sample_data[col][0],str):
            for scan in params['Samples'][sample]['Sample scans']:
                if int(sample_data[col][0].split(' / ')[1].split(' ')[1]) == scan:
                    
                    sample_power[f"Scan_{str(scan)}"]['Temperature'] = np.array(sample_data[col][2:].astype(float))
                    sample_power[f"Scan_{str(scan)}"]['Raw_power'] = np.array(sample_data[f"Unnamed: {x+1}"][2:].astype(float))*1e-6
                    
                    try:
                        sample_power[f"Scan_{str(scan)}"]['Buffer_subtracted_power'] = sample_power[f"Scan_{str(scan)}"]['Raw_power'] - mean_buffer_power
                        
                    except:
                        if len(sample_power[f"Scan_{str(scan)}"]['Raw_power']) > len(mean_buffer_power):
                            sample_power[f"Scan_{str(scan)}"]['Raw_power'] = sample_power[f"Scan_{str(scan)}"]['Raw_power'][:-1]
                            sample_power[f"Scan_{str(scan)}"]['Buffer_subtracted_power'] = sample_power[f"Scan_{str(scan)}"]['Raw_power'] - mean_buffer_power
                            sample_power[f"Scan_{str(scan)}"]['Temperature'] = sample_power[f"Scan_{str(scan)}"]['Temperature'][:-1]
                            
                        elif len(sample_power[f"Scan_{str(scan)}"]['Raw_power']) < len(mean_buffer_power):
                            mean_buffer_temperature = mean_buffer_temperature[:-1]
                            mean_buffer_power = mean_buffer_power[:-1]

                    # Plot sample power data
                    axs[0].plot(sample_power[f"Scan_{str(scan)}"]['Temperature'][100:],sample_power[f"Scan_{str(scan)}"]['Raw_power'][100:],LineWidth=2,label=f"Sample scan {str(scan)}")
                    axs[0].plot(sample_power[f"Scan_{str(scan)}"]['Temperature'][100:],sample_power[f"Scan_{str(scan)}"]['Buffer_subtracted_power'][100:],LineWidth=2,label='Buffer subtracted sample scan')
                    axs[0].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
                    axs[0].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
                    axs[0].yaxis.major.formatter._useMathText = True
                    #axs[0].set_title(f"{sample} Scan {str(scan)}",fontsize=14)
                    axs[0].set_xlabel('Temperature \N{DEGREE SIGN}C',fontsize=14)
                    axs[0].set_ylabel('Power J $s^{-1}$',fontsize=14)
                    axs[0].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
                    xticks = np.arange(np.round(np.min(sample_power[f"Scan_{str(scan)}"]['Temperature'])),np.round(np.nanmax(sample_power[f"Scan_{str(scan)}"]['Temperature']))+1,5)
                    axs[0].set_xticks(xticks)
                    axs[0].legend(loc='upper left',frameon=False,fontsize=12)

                    # Calculate water heat capacity and density as a function of temperature
                    Cp_water, p_water = WaterProperties(sample_power[f"Scan_{str(scan)}"]['Temperature'])
                    
                    # Calculate sample heat capacity                    
                    # Scan rate is in degrees per second
                    SR = params['Samples'][sample]['Scan rate']*(1/60)
                    # MW in g/mol
                    MW = params['Samples'][sample]['Molecular weight']
                    # Concentration in mg/mL
                    CT = params['Samples'][sample]['Concentration']*1e-6*params['Samples'][sample]['Molecular weight']
                    
                    sample_Cp[f"Scan_{str(scan)}"]['Temperature'] = sample_power[f"Scan_{str(scan)}"]['Temperature']
                    sample_Cp[f"Scan_{str(scan)}"]['Cp'] = ((sample_power[f"Scan_{str(scan)}"]['Buffer_subtracted_power']/SR)/(CT*Vcell) + Cp_water*0.73*p_water)*(MW/1000)
                    
                    axs[1].plot(sample_Cp[f"Scan_{str(scan)}"]['Temperature'][100:],sample_Cp[f"Scan_{str(scan)}"]['Cp'][100:],LineWidth=2,label=f'Sample scan {str(scan)}')
                    #axs[1].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
                    axs[1].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
                    axs[1].yaxis.major.formatter._useMathText = True
                    axs[1].set_title(f"{sample}",fontsize=14)
                    axs[1].set_xlabel('Temperature \N{DEGREE SIGN}C',fontsize=14)
                    axs[1].set_ylabel('$C_{p}$ kJ $mol^{-1}$ $K^{-1}$',fontsize=14)
                    axs[1].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
                    xticks = np.arange(np.round(np.min(sample_Cp[f"Scan_{str(scan)}"]['Temperature'])),np.round(np.nanmax(sample_Cp[f"Scan_{str(scan)}"]['Temperature']))+1,5)
                    axs[1].set_xticks(xticks)
                    axs[1].legend(loc='upper left',frameon=False,fontsize=12)
                    
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close()
                    #axs[0].set_ylabel('$C_{p}$ $kJ$ $mol^{-1}$ $K^{-1}$',fontsize=14)
                    pickle_out = open(params['Samples'][sample]['Save'],"wb")
                    pickle.dump(sample_Cp, pickle_out)
                    pickle_out.close()
                    
pdf.close()
#plt.show()
                    
    
    