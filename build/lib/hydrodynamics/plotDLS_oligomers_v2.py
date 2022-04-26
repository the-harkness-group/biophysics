#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:02:45 2021

@author: robertharkness
"""

import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import customcolormap
import hydrodynamics
import copy
import matplotlib.ticker

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

def main():
    
    # Read parameters and dataset to be plotted
    params = yaml.safe_load(open(sys.argv[1],'r'))

    ######################### DATA ################################
    ### Datasets, just need to be in /Manuscripts/CryoCages/Figures/Fig2 and all data paths are defined relative to this
    # P2 + 15N hTRF1
    data1 = pd.read_csv('../../../../../DLS/202107/20210709/DegP2_RF_HSDLS_SubstrateBinding_Tramp_20210709_fitparams.csv')
    # P2 + TrpCage-hTRF1
    data2 = pd.read_csv('../../../../../DLS/202105/20210514/DegP2_RF_HSDLS_TrpCagehTRF1_Binding_Tramp_fitparams_20210514.csv')
    # P2 + His-SUMO-hTRF1
    data3 = pd.read_csv('../../../../../DLS/202107/20210708/PepD_DegP2_substratebinding_Tramp_20210708_fitparams.csv')
    # P2 + IL6-hTRF1
    data4 = pd.read_csv('../../../../../DLS/202107/20210702/DegP2_RF_HSDLS_SubstrateBinding_Tramp_20210702_fitparams.csv')
    # P2 + GFP-hTRF1
    data5 = pd.read_csv('../../../../../DLS/202105/20210503/DegP2_RF_HSDLS_clientbinding_Tramp_fitparams_20210503.csv')
    # P2 + IL6 and MNeon, no hTRF1 tag
    data6 = pd.read_csv('../../../../../DLS/202107/20210722/DegP2_RF_SubstrateBinding_DegP38_hTRF1Binding_Tramp_20210722_fitparams.csv')

    ### Apo and P2-bound data for each substrate
    # Apo P2
    apo_P2 = data1[data1['Well'] == 'C19']
    # hTRF1 15N
    apo_hTRF1 = data3[data3['Well'] == 'B24']
    bound_hTRF1 = data1[data1['Well'] == 'C16']
    # TrpCage-hTRF1
    apo_TrpCage = data3[data3['Well'] == 'C1']
    bound_TrpCage = data2[data2['Well'] == 'B24']
    # HS-hTRF1
    apo_HS = data3[data3['Well'] == 'B22']
    bound_HS = data3[data3['Well'] == 'B18']
    # IL6-hTRF1
    apo_IL6 = data3[data3['Well'] == 'C2']
    bound_IL6 = data4[data4['Well'] == 'A3']
    # MNeon-hTRF1
    apo_MNeon = data3[data3['Well'] == 'B23']
    bound_MNeon = data5[data5['Well'] == 'B19']

    ### Negative controls, P2 + substrates with no hTRF1 tag
    # TrpCage
    # HS
    apo_neg_HS = data4[data4['Well'] == 'A11']
    bound_neg_HS = data4[data4['Well'] == 'A6']
    # IL6
    apo_neg_IL6 = data6[data6['Well'] == 'D5']
    bound_neg_IL6 = data6[data6['Well'] == 'C22']
    # MNeon
    apo_neg_MNeon = data6[data6['Well'] == 'D6']
    bound_neg_MNeon = data6[data6['Well'] == 'D1']
    #############################################################################

    ### Substrate chain lengths
    length = [0,54,75,164,239,289]

    ### Calculate oligomer D lines at desired buffer viscosity as a fxn of T
    Temperatures = np.linspace(5,40,100) + 273.15
    D_dict = {'D3':np.zeros(len(Temperatures)),'D6A':np.zeros(len(Temperatures)),
              'D12A':np.zeros(len(Temperatures)),'D18A':np.zeros(len(Temperatures)),
              'D24':np.zeros(len(Temperatures)),'D30':np.zeros(len(Temperatures)),
              'D60':np.zeros(len(Temperatures))}
    eta = hydrodynamics.viscosity(Temperatures, params['Eta params']) # Get diffusion coeffs as a fxn of temperature
    Do = hydrodynamics.stokes_diffusion(Temperatures, eta, float(params['Trimer Rh']))
    D_dict['D3'] = hydrodynamics.scaled_diffusion(Do, 1, -0.333)        
    D_dict['D6A'] = hydrodynamics.scaled_diffusion(Do, 2, -0.227)
    D_dict['D12A'] = hydrodynamics.scaled_diffusion(Do, 4, -0.333)
    #D_dict['D18A'] = hydrodynamics.scaled_diffusion(Do, 6, -0.333)
    D_dict['D24'] = hydrodynamics.scaled_diffusion(Do, 8, -0.333)
    #D_dict['D30'] = hydrodynamics.scaled_diffusion(Do, 10, -0.333)
    D_dict['D60'] = hydrodynamics.scaled_diffusion(Do, 20, -0.333)

    ### Calculate oligomer D lines at ruler fig T
    ruler_Temp = 37.5
    ruler_dict = {'D3':np.zeros(1),'D6A':np.zeros(1),'D12A':np.zeros(1),'D24':np.zeros(1),'D60':np.zeros(1)}
    eta = hydrodynamics.viscosity(ruler_Temp+273.15, params['Eta params']) # Get diffusion coeffs as a fxn of temperature
    Do = hydrodynamics.stokes_diffusion(ruler_Temp+273.15, eta, float(params['Trimer Rh']))
    ruler_dict['D3'] = hydrodynamics.scaled_diffusion(Do, 1, -0.333)        
    ruler_dict['D6A'] = hydrodynamics.scaled_diffusion(Do, 2, -0.227)
    ruler_dict['D12A'] = hydrodynamics.scaled_diffusion(Do, 4, -0.333)
    ruler_dict['D24'] = hydrodynamics.scaled_diffusion(Do, 8, -0.333)
    ruler_dict['D60'] = hydrodynamics.scaled_diffusion(Do, 20, -0.333)

    ### Calculate substrate D lines using D at 5C
    subs_dict = {'D_hTRF1':np.zeros(len(Temperatures)),'D_TrpCage':np.zeros(len(Temperatures)),'D_HS':np.zeros(len(Temperatures)),
    'D_IL6':np.zeros(len(Temperatures)),'D_MNeon':np.zeros(len(Temperatures)),'Rh_hTRF1':np.zeros(1),
    'Rh_TrpCage':np.zeros(1),'Rh_HS':np.zeros(1),'Rh_IL6':np.zeros(1),
    'Rh_MNeon':np.zeros(1)}
    eta = hydrodynamics.viscosity(5+273.15, params['Eta params']) # Get eta at 5C
    subs_dict['Rh_hTRF1'] = hydrodynamics.stokes_radius(5+273.15,eta,apo_hTRF1[apo_hTRF1['Temperature'] == 5].D.values)
    subs_dict['Rh_TrpCage'] = hydrodynamics.stokes_radius(5+273.15,eta,apo_TrpCage[apo_TrpCage['Temperature'] == 5].D.values)
    subs_dict['Rh_HS'] = hydrodynamics.stokes_radius(5+273.15,eta,apo_HS[apo_HS['Temperature'] == 5].D.values)
    subs_dict['Rh_IL6'] = hydrodynamics.stokes_radius(5+273.15,eta,apo_IL6[apo_IL6['Temperature'] == 5].D.values)
    subs_dict['Rh_MNeon'] = hydrodynamics.stokes_radius(5+273.15,eta,apo_MNeon[apo_MNeon['Temperature'] == 5].D.values)

    print(subs_dict['Rh_hTRF1'])
    print(subs_dict['Rh_TrpCage'])
    print(subs_dict['Rh_HS'])
    print(subs_dict['Rh_IL6'])
    print(subs_dict['Rh_MNeon'])

    eta = hydrodynamics.viscosity(Temperatures, params['Eta params']) # Get eta at all temperatures
    subs_dict['D_hTRF1'] = hydrodynamics.stokes_diffusion(Temperatures, eta, float(subs_dict['Rh_hTRF1']))
    subs_dict['D_TrpCage'] = hydrodynamics.stokes_diffusion(Temperatures, eta, float(subs_dict['Rh_TrpCage']))
    subs_dict['D_HS'] = hydrodynamics.stokes_diffusion(Temperatures, eta, float(subs_dict['Rh_HS']))
    subs_dict['D_IL6'] = hydrodynamics.stokes_diffusion(Temperatures, eta, float(subs_dict['Rh_IL6']))
    subs_dict['D_MNeon'] = hydrodynamics.stokes_diffusion(Temperatures, eta, float(subs_dict['Rh_MNeon']))

    ### Set up plotting
    plt.style.use('figure')
    apo_fig, apo_ax = plt.subplots(1,1)
    bound_fig, bound_ax = plt.subplots(1,1)
    apo_ax.set_xlabel('Temperature \N{DEGREE SIGN}C')
    apo_ax.set_ylabel('$\it{D_{z}}$ cm$^{2}$ s$^{-1}$')
    bound_ax.set_xlabel('Temperature \N{DEGREE SIGN}C')
    bound_ax.set_ylabel('$\it{D_{z}}$ cm$^{2}$ s$^{-1}$')
    ruler_fig, ruler_ax = plt.subplots(1,1,figsize=(4,4.8))
    ruler_ax.set_xlabel('Client length')
    ruler_ax.set_ylabel('$\it{D_{z}}$ cm$^{2}$ s$^{-1}$')
    neg_fig, neg_ax = plt.subplots(1,1)
    neg_ax.set_xlabel('Temperature \N{DEGREE SIGN}C')
    neg_ax.set_ylabel('$\it{D_{z}}$ cm$^{2}$ s$^{-1}$')

    # Black-pink colormap
    hex_colors = ['#020101','#281112','#4E2128','#733243','#984364','#BD548A','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF']
    hex_colors = list(reversed(hex_colors))
    # Oligomer D line colors
    #M3_color = '#2b8cbe'
    #M6_color = '#fc8d59'
    #M12_color = '#9e0142'
    #M18_color = '#8856a7'
    #M24_color = '#78c679'
    #M30_color = '#dd3497'
    #M60_color = 'k'

    M60_color = 'k' # Black lines
    M24_color = 'k'
    M12_color = 'k'
    M6_color = 'k'
    M3_color = 'k'

    # Apo P2 color
    c_P2 = '#dd3497'
    # Substrate colors
    c_hTRF1 = '#D2B48C'
    c_TrpCage = '#87CEEB'
    c_HS = '#7B68EE'
    c_IL6 = '#FA8072'
    c_MNeon = '#90EE90'
    
    ### Apo substrate plot
    # D lines estimated from substrate Rh calculated at 5C
    apo_ax.plot(Temperatures-273.15,subs_dict['D_hTRF1']*1e4,'--',linewidth=2,color=M3_color,alpha=0.4)
    apo_ax.plot(Temperatures-273.15,subs_dict['D_TrpCage']*1e4,'--',linewidth=2,color=M3_color,alpha=0.4)
    apo_ax.plot(Temperatures-273.15,subs_dict['D_HS']*1e4,'--',linewidth=2,color=M3_color,alpha=0.4)
    apo_ax.plot(Temperatures-273.15,subs_dict['D_IL6']*1e4,'--',linewidth=2,color=M3_color,alpha=0.4)
    apo_ax.plot(Temperatures-273.15,subs_dict['D_MNeon']*1e4,'--',linewidth=2,color=M3_color,alpha=0.4)

    # Data
    apo_ax.plot(apo_hTRF1.Temperature,apo_hTRF1.D*1e4,'o',color=c_hTRF1)
    apo_ax.plot(apo_TrpCage.Temperature,apo_TrpCage.D*1e4,'o',color=c_TrpCage)
    apo_ax.plot(apo_HS.Temperature,apo_HS.D*1e4,'*',markersize='12',color=c_HS,mec='k',mew='0.75') # Star marker because close to IL6 in Dz
    apo_ax.plot(apo_IL6.Temperature,apo_IL6.D*1e4,'o',color=c_IL6)
    apo_ax.plot(apo_MNeon.Temperature,apo_MNeon.D*1e4,'o',color=c_MNeon)

    ### P2 + bound substrate plot
    # Oligomer lines
    bound_ax.plot(Temperatures-273.15,D_dict['D3']*1e4,'--',linewidth=2,color=M3_color,alpha=0.4)
    bound_ax.plot(Temperatures-273.15,D_dict['D6A']*1e4,'--',linewidth=2,color=M6_color,alpha=0.4)
    bound_ax.plot(Temperatures-273.15,D_dict['D12A']*1e4,'--',linewidth=2,color=M12_color,alpha=0.4)
    bound_ax.plot(Temperatures-273.15,D_dict['D24']*1e4,'--',linewidth=2,color=M24_color,alpha=0.4)
    bound_ax.plot(Temperatures-273.15,D_dict['D60']*1e4,'--',linewidth=2,color=M60_color,alpha=0.4)

    # Data
    bound_ax.plot(apo_P2.Temperature,apo_P2.D*1e4,'o',color=c_P2)
    bound_ax.plot(bound_hTRF1.Temperature,bound_hTRF1.D*1e4,'d',markersize='8',color=c_hTRF1,mec='k',mew='0.75')
    bound_ax.plot(bound_TrpCage.Temperature,bound_TrpCage.D*1e4,'o',color=c_TrpCage)
    bound_ax.plot(bound_HS.Temperature,bound_HS.D*1e4,'o',color=c_HS)
    bound_ax.plot(bound_IL6.Temperature,bound_IL6.D*1e4,'o',color=c_IL6)
    bound_ax.plot(bound_MNeon.Temperature,bound_MNeon.D*1e4,'o',color=c_MNeon)

    ### Substrate ruler plot
    # Oligomer lines
    ruler = np.linspace(0,300,100)
    ruler_ax.plot(ruler,[ruler_dict['D12A']*1e4 for x in range(len(ruler))],'--',color=M12_color,alpha=0.4,zorder=0)
    ruler_ax.plot(ruler,[ruler_dict['D24']*1e4 for x in range(len(ruler))],'--',color=M24_color,alpha=0.4,zorder=0)
    ruler_ax.plot(ruler,[ruler_dict['D60']*1e4 for x in range(len(ruler))],'--',color=M60_color,alpha=0.4,zorder=0)

    # Data
    ruler_ax.bar(length[0],apo_P2[apo_P2.Temperature.values==ruler_Temp].D*1e4,width=22,color=c_P2)
    ruler_ax.bar(length[1],bound_hTRF1[bound_hTRF1.Temperature.values==ruler_Temp].D*1e4,width=22,color=c_hTRF1)
    ruler_ax.bar(length[2],bound_TrpCage[bound_TrpCage.Temperature.values==ruler_Temp].D*1e4,width=22,color=c_TrpCage)
    ruler_ax.bar(length[3],bound_HS[bound_HS.Temperature.values==ruler_Temp].D*1e4,width=22,color=c_HS)
    ruler_ax.bar(length[4],bound_IL6[bound_IL6.Temperature.values==ruler_Temp].D*1e4,width=22,color=c_IL6)
    ruler_ax.bar(length[5],bound_MNeon[bound_MNeon.Temperature.values==ruler_Temp].D*1e4,width=22,color=c_MNeon)

    ### Negative substrate without hTRF1 tag control plot
    # Oligomer D lines
    neg_ax.plot(Temperatures-273.15,D_dict['D3']*1e4,'--',linewidth=2,color=M3_color,alpha=0.4)
    neg_ax.plot(Temperatures-273.15,D_dict['D6A']*1e4,'--',linewidth=2,color=M6_color,alpha=0.4)
    neg_ax.plot(Temperatures-273.15,D_dict['D12A']*1e4,'--',linewidth=2,color=M12_color,alpha=0.4)
    neg_ax.plot(Temperatures-273.15,D_dict['D24']*1e4,'--',linewidth=2,color=M24_color,alpha=0.4)
    neg_ax.plot(Temperatures-273.15,D_dict['D60']*1e4,'--',linewidth=2,color=M60_color,alpha=0.4)

    # Data
    neg_ax.plot(apo_P2.Temperature,apo_P2.D*1e4,'o',color=c_P2)
    #neg_ax.plot(neg_TrpCage.Temperature,neg_TrpCage.D*1e4,'o',color=c_TrpCage)
    neg_ax.plot(bound_neg_HS.Temperature,bound_neg_HS.D*1e4,'o',color=c_HS)
    neg_ax.plot(bound_neg_IL6.Temperature,bound_neg_IL6.D*1e4,'o',color=c_IL6)
    neg_ax.plot(bound_neg_MNeon.Temperature,bound_neg_MNeon.D*1e4,'o',color=c_MNeon)

    # Axis settings
    apo_ax.set_xlim([3.5,41.5])
    apo_ax.set_ylim([0.2e-6,2.35e-6])
    apo_ax.set_yticks([0.3e-6,1.3e-6,2.3e-6])
    bound_ax.set_xlim([3.5,41.5])
    bound_ax.set_ylim([0.7e-7,5.2e-7])
    bound_ax.set_yticks([1e-7,3e-7,5e-7])
    neg_ax.set_xlim([3.5,41.5])
    neg_ax.set_ylim([2.0e-7,6.0e-7])
    neg_ax.set_yticks([2e-7,4e-7,6e-7])
    ruler_ax.set_xticks([0,150,300])

    ax = [apo_ax,bound_ax,ruler_ax,neg_ax]
    for axe in ax:
        axe.yaxis.set_major_formatter(OOMFormatter(-6, "%1.1f"))
        axe.ticklabel_format(axis='y', style='sci', scilimits=(-6,-6))

    apo_fig.tight_layout()
    bound_fig.tight_layout()
    ruler_fig.tight_layout()
    neg_fig.tight_layout()

    apo_fig.savefig('Apo_fig.pdf',format='pdf')
    bound_fig.savefig('Bound_fig.pdf',format='pdf')
    ruler_fig.savefig('Ruler_fig.pdf',format='pdf')
    neg_fig.savefig('Negative_control_fig.pdf',format='pdf')

main()
    
    