#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:23:04 2021

@author: robertharkness
"""

import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bindingmodels
import linearextensionto12mer
import linearextensionto24mer
import linearextensionto36mer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from lmfit import minimize, Parameters, report_fit

def calcfractionbound():
    
    params= yaml.safe_load(open(sys.argv[1],'r')) # Get data for P2, P9, P33
    P2_files = params['Sample']['P2']['Files']
    P2_DSS = pd.read_csv(params['Sample']['P2']['DSS'][0],delim_whitespace=True)
    P2_DSS.columns = ['Assignment','w1','w2','w1 Hz','w2 Hz','dummy','dummy']
    P9_data = pd.read_csv(params['Sample']['P9']['Files'][0],delim_whitespace=True)
    P9_data.columns = ['Assignment','w1','w2','w1 Hz','w2 Hz','dummy','dummy']
    P9_DSS = pd.read_csv(params['Sample']['P9']['DSS'][0],delim_whitespace=True)
    P9_DSS.columns = ['Assignment','w1','w2','w1 Hz','w2 Hz','dummy','dummy']
    P33_fitparams = pd.read_csv(params['Sample']['P33']['Parameters'][0])
    P33_fitdata = pd.read_csv(params['Sample']['P33']['Fit'][0])
    P33_exptdata = pd.read_csv(params['Sample']['P33']['Experiment'][0])
    
    # For comparison with PDZ1 U-2H, ILVM titration at 50C
    P33_exptgroups = P33_exptdata.groupby('Assignment',sort=False)
    P33_fitgroups = P33_fitdata.groupby('Assignment',sort=False)
    P33_exptdict = {'L272CD1-HD1':[],'L272CD2-HD2':[],
               'L276CD1-HD1':[],'L276CD2-HD2':[],
               'M280CE-HE':[]}
    P33_fitdict = {'L272CD1-HD1':[],'L272CD2-HD2':[],
               'L276CD1-HD1':[],'L276CD2-HD2':[],
               'M280CE-HE':[]}
    P33_mean = []
    P33_expt_LT = np.array(P33_exptgroups.get_group(list(P33_exptgroups.groups)[0]).LT.values)
    P33_fit_LT = np.array(P33_fitgroups.get_group(list(P33_fitgroups.groups)[0]).LT.values)
    
    for k in P33_exptdict.keys():
        P33_exptdict[k].append(P33_exptdata[P33_exptdata['Assignment'] == k]['fb_expt'].values)
        P33_fitdict[k].append(P33_fitdata[P33_fitdata['Assignment'] == k]['fb_fit'].values)
        P33_mean.append(P33_fitdata[P33_fitdata['Assignment'] == k]['fb_fit'].values)
    P33_mean = np.mean(P33_mean,0)
            
    # Dictionary of P2 residues for calculating dw
    P2_dict = {'L272CD1-HD1':{'1H':[],'13C':[],'FB':[]},'L272CD2-HD2':{'1H':[],'13C':[],'FB':[]},
               'L276CD1-HD1':{'1H':[],'13C':[],'FB':[]},'L276CD2-HD2':{'1H':[],'13C':[],'FB':[]},
               'M280CE-HE':{'1H':[],'13C':[],'FB':[]}}
    
    # Iterate over P2 files and calculate dw, use internal referencing
    # THIS IS CURRENTLY REFERENCING RELATIVE TO THE TRIMER INTERNAL REFERENCE PEAK
    P9_ref_1H = P9_data[P9_data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0] # 2021-05-10 trimer internal reference peak
    P9_ref_13C = P9_data[P9_data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0]
    
    for index, file in enumerate(P2_files):
        
        data = pd.read_csv(file,delim_whitespace=True)
        data.columns = ['Assignment','w1','w2','w1 Hz','w2 Hz','dummy','dummy']
        
        dr_1H = P9_ref_1H - data[data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0] # Shift each peak below by adding this delta = (M3 IR - peak,i)
        dr_13C = P9_ref_13C - data[data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0]
        
        for k in P2_dict.keys():
            
            # THIS BLOCK OF CODE IS FOR REFERENCING RELATIVE TO FIRST P2 POINT
            #if params['Internal reference'] == 'y':
                #if index == 0:
                    #ref_1H = data[data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0] # Get internal reference frequency
                    #ref_13C = data[data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0]
                    
                    #P9_shift_1H = ref_1H - P9_data[P9_data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0] # Calculate internal ref for trimer against first P2 file
                    #P9_shift_13C = ref_13C - P9_data[P9_data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0] # 2021-05-10
                    
                #P9_corr_1H = P9_data[P9_data['Assignment'] == k]['w2 Hz'].values[0] + P9_shift_1H # Get corrected shift for each trimer residue
                #P9_corr_13C = P9_data[P9_data['Assignment'] == k]['w1 Hz'].values[0] + P9_shift_13C
                
                #dr_1H = ref_1H - data[data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0] # Get shift for each WT residue relative to first P2 point
                #dr_13C = ref_13C - data[data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0]
                
                # Calculate dw and use internal peak as reference
                #P2_dict[k]['1H'].append(float(data[data['Assignment'] == k]['w2 Hz'].values[0]) + dr_1H # Shift with internal reference
                #       - P9_corr_1H) # Subtract corrected trimer as endpoint
                
                #P2_dict[k]['13C'].append(float(data[data['Assignment'] == k]['w1 Hz'].values[0]) + dr_13C
                #       - P9_corr_13C)
            
            # REFERENCE EACH SO THAT ALL INTERNAL REFERENCE PEAKS HAVE SAME SHIFT, B' = B + (A - B) = A, 2021-05-10
            P2_dict[k]['1H'].append(float(data[data['Assignment'] == k]['w2 Hz'].values[0]) + dr_1H # Shift with internal reference to trimer
                       - P9_data[P9_data['Assignment'] == k]['w2 Hz'].values[0]) # Subtract trimer as w,0, so dw = (w,i - w,0) 2021-05-10
               
            P2_dict[k]['13C'].append(float(data[data['Assignment'] == k]['w1 Hz'].values[0]) + dr_13C
                       - P9_data[P9_data['Assignment'] == k]['w1 Hz'].values[0])
            
            # WITH CORRECT REFERENCING ALL INTERNAL REFERENCE PEAKS SHOULD HAVE SAME W1 AND SAME W2!
            #print(P9_data[P9_data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0] + P9_shift_1H)
            #print(P9_data[P9_data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0] + P9_shift_13C)
            print('TRIMER INTERNAL REFERENCE, w2, w1')
            print(P9_data[P9_data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0])
            print(P9_data[P9_data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0])
            print('S210A INTERNAL REFERENCE, w2, w1')
            print(float(data[data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0]) + dr_1H)
            print(float(data[data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0]) + dr_13C)
    
    # Calculate model-free fraction bound
    for k in P2_dict.keys():
        
        P2_dict[k]['FB'] = np.sqrt(np.array(P2_dict[k]['1H'])**2 + np.array(P2_dict[k]['13C'])**2)/P33_fitparams[f"dw_{k.split('-')[0] + k.split('-')[1]}"].values[0]
        print(f"FB for {k} is: {P2_dict[k]['FB']}")
    
    # Simulate fraction bound to 12-mer for 1 PDZ contact
    #P_dict, C_dict = bindingmodels.make_dictionaries(['3A','6B','9B','12B'])
    #P_dict, C_dict = bindingmodels.make_dictionaries(['3A','6B','9B','12B','15B','18B','21B','24B'])
    P_dict, C_dict = bindingmodels.make_dictionaries(['3A','6B','9B','12B','15B','18B','21B','24B', '27B', '30B', '33B', '36B'])
    #sim_func = linearextensionto24mer.linear_extension
    #sim_func_fb = linearextensionto24mer.fractionbound
    #sim_func = linearextensionto12mer.linear_extension
    #sim_func_fb = linearextensionto12mer.fractionbound
    sim_func = linearextensionto36mer.linear_extension
    sim_func_fb = linearextensionto36mer.fractionbound
    #fit_params = {'K1':1/946e-6,'cooperativity':1}
    fit_constants = {'model':'2'}
    
    # Fit fraction bound, make df first
    fit_dict = {'Assignment':[],'FB':[],'Temperature':[],'Concentration':[]}
    for k in P2_dict.keys():
        for index, concentration in enumerate(params['Sample']['P2']['Concentrations']):
            fit_dict['Assignment'].append(k)
            fit_dict['FB'].append(P2_dict[k]['FB'][index])
            fit_dict['Concentration'].append(concentration)
            fit_dict['Temperature'].append(50)
    fit_df = pd.DataFrame(fit_dict)
#    fit_df.to_csv(params['Filename'].split('.')[0] + '.csv')
    
    fit_params = Parameters()
    fit_params.add('K1',value=1/946e-6)
    fit_params.add('cooperativity',value=1,vary=False)
    
    result = minimize(objective, fit_params, method='nelder', args=(fit_df, 
        P_dict, C_dict, fit_constants, sim_func, sim_func_fb))
    report_fit(result)
    
    FB_1 = []
    for concentration in P33_fit_LT:
        P_dict, C_dict = sim_func(result.params, fit_constants, 50+273.15, concentration*1e-6, P_dict, C_dict)
        FB_1.append(sim_func_fb(P_dict, fit_params, fit_constants['model']))    
    
#    FB_1 = []
#    for concentration in P33_fit_LT:
#        P_dict, C_dict = sim_func(fit_params, fit_constants, 50+273.15, concentration*1e-6, P_dict, C_dict)
#        FB_1.append(sim_func_fb(P_dict, fit_params, fit_constants['model']))
#    
#    # Simulate fraction bound to 12-mer for 2 PDZ contacts
#    fit_constants['model'] = '2'
#    FB_2 = []
#    for concentration in P33_fit_LT:
#        P_dict, C_dict = sim_func(fit_params, fit_constants, 50+273.15, concentration*1e-6, P_dict, C_dict)
#        FB_2.append(sim_func_fb(P_dict, fit_params, fit_constants['model']))
#    
#    fit_constants['model'] = '2'
#    fit_params['cooperativity'] = 10 # Positive cooperativity
#    FB_2_coop = []
#    for concentration in P33_fit_LT:
#        P_dict, C_dict = sim_func(fit_params, fit_constants, 50+273.15, concentration*1e-6, P_dict, C_dict)
#        FB_2_coop.append(sim_func_fb(P_dict, fit_params, fit_constants['model']))
#        
    # Upper limit for all 3 bound PDZ
    FB_3 = [1.0 for x in range(len(P33_fit_LT))]
    
    plt.style.use('figure')
    fig, ax = plt.subplots(1,1,figsize=(11,8.64))
    axins = inset_axes(ax,width="30%",height="30%",loc=2,bbox_to_anchor=(0.065,0.01,0.995,0.995), bbox_transform=ax.transAxes)
    colors = ['#74add1','#abdda4','#9970ab','#fdae61','#de77ae']
    colors = colors[::-1]
    ax.plot(P33_fit_LT/1000,P33_mean,'k--',linewidth=4)
    axins.plot(P33_fit_LT/1000,P33_mean,'k--',linewidth=4)
    for index, k in enumerate(P33_exptdict.keys()):
        ax.plot(P33_expt_LT/1000,np.ravel(P33_exptdict[k]),'d',markersize=16,color=colors[index])
        axins.plot(P33_expt_LT/1000,np.ravel(P33_exptdict[k]),'d',markersize=16,color=colors[index])
    for index, k in enumerate(P2_dict.keys()):
        ax.plot(np.array(params['Sample']['P2']['Concentrations'])/1000,np.ravel(P2_dict[k]['FB']),'o',color=colors[index],markersize=12)
        axins.plot(np.array(params['Sample']['P2']['Concentrations'])/1000,np.ravel(P2_dict[k]['FB']),'o',color=colors[index],markersize=12)
        
        print('Methyl is:',k,'Color is:',colors[index])
        
    ax.plot(P33_fit_LT/1000,FB_1,'--',color='#d53e4f',linewidth=4)
    #ax.plot(P33_fit_LT/1000,FB_2,'--',color='#f46d43',linewidth=4)
    #ax.plot(P33_fit_LT/1000,FB_2_coop,'--',color='#66bd63',linewidth=4)
    #ax.plot(P33_fit_LT/1000,FB_3,'--',color='#80cdc1',linewidth=4)
    #ax.plot(params['Sample']['P2']['Concentrations'],result.best_fit,'ko',markersize=12)
    ax.set_xlabel('$\it{M_{T,S210A}}$ or [PDZ2] mM',fontsize=48)
    ax.set_ylabel('Fraction bound',fontsize=48)
    ax.set_ylim([-0.03,1.03])
    ax.set_xlim([np.min(P33_fit_LT/1000)-0.03*np.max(P33_fit_LT/1000),np.max(P33_fit_LT/1000)+0.005*np.max(P33_fit_LT/1000)])
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(['0','1','2','3','4','5'],fontsize=36)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(['0','0.2','0.4','0.6','0.8','1.0'],fontsize=36)
    axins.set_xticks([0,0.1,0.2])
    axins.set_yticks([0.0,0.1,0.2,0.30])
    axins.set_xlim([-0.015, 0.25])
    axins.set_ylim([-0.03,0.39])
    axins.yaxis.major.formatter._useMathText = True
    for item in (axins.get_xticklabels() + axins.get_yticklabels()):
        item.set_fontsize(30)
    
    fig.tight_layout()
    fig.savefig('DegP2_modelfree_fractionbound.pdf',format='pdf')
    plt.close(fig)

def objective(fit_params, fit_df, P_dict, C_dict, fit_constants, sim_func,
    sim_func_fb):
    
    residuals = []
    for i,v in enumerate(fit_df['FB']):
        
        P_dict, C_dict = sim_func(fit_params, fit_constants, fit_df['Temperature'].iloc[i]+273.15, 
                                  fit_df['Concentration'].iloc[i]*1e-6, P_dict, C_dict)
        FB_calc = sim_func_fb(P_dict, fit_params, fit_constants['model'])
        
        residuals.append(fit_df['FB'].iloc[i] - FB_calc)
    
    return residuals

calcfractionbound()
