#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:30:13 2021

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
import linearextensionto24mer_2PDZ1PDZ2
import linearextensionto36mer
import linearextensiontonmer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from lmfit import minimize, Parameters, report_fit
import pickle

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
    P33_expt_mean = []
    P33_expt_LT = np.array(P33_exptgroups.get_group(list(P33_exptgroups.groups)[0]).LT.values)
    P33_fit_LT = np.array(P33_fitgroups.get_group(list(P33_fitgroups.groups)[0]).LT.values)
    
    for k in P33_exptdict.keys():
        P33_exptdict[k].append(P33_exptdata[P33_exptdata['Assignment'] == k]['fb_expt'].values)
        P33_fitdict[k].append(P33_fitdata[P33_fitdata['Assignment'] == k]['fb_fit'].values)
        P33_mean.append(P33_fitdata[P33_fitdata['Assignment'] == k]['fb_fit'].values)
        P33_expt_mean.append(P33_exptdata[P33_exptdata['Assignment'] == k]['fb_expt'].values)
    P33_mean = np.mean(P33_mean,0)
    P33_expt_mean = np.mean(P33_expt_mean,0)
            
    # Dictionary of P2 residues for calculating dw
    P2_dict = {'L272CD1-HD1':{'1H':[],'13C':[],'FB':[]},'L272CD2-HD2':{'1H':[],'13C':[],'FB':[]},
               'L276CD1-HD1':{'1H':[],'13C':[],'FB':[]},'L276CD2-HD2':{'1H':[],'13C':[],'FB':[]},
               'M280CE-HE':{'1H':[],'13C':[],'FB':[]}}
    
    # Iterate over P2 files and calculate dw, use internal referencing
    # THIS IS CURRENTLY REFERENCING RELATIVE TO THE TRIMER INTERNAL REFERENCE PEAK
    P9_ref_1H = P9_data[P9_data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0] # 2021-05-10 trimer internal reference peak
    P9_ref_13C = P9_data[P9_data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0]
    
    # Check referencing with a 2D plot
    chfig,chax = plt.subplots(1,1)
    chax.set_xlabel('w2 Hz')
    chax.set_ylabel('w1 Hz')
    chax.plot(P9_ref_1H,P9_ref_13C,'ro',markersize=6) # Plot trimer internal reference
    chax.set_ylim([5250,3250])
    chax.set_xlim([1800,200])
    
    for index, file in enumerate(P2_files):
        
        data = pd.read_csv(file,delim_whitespace=True)
        data.columns = ['Assignment','w1','w2','w1 Hz','w2 Hz','dummy','dummy']
        
        dr_1H = P9_ref_1H - data[data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0] # Shift each peak below by adding this delta = (M3 IR - peak,i)
        dr_13C = P9_ref_13C - data[data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0]
        
        for k in P2_dict.keys():
            
            # REFERENCE EACH SO THAT ALL INTERNAL REFERENCE PEAKS HAVE SAME SHIFT, B' = B + (A - B) = A, 2021-05-10
            P2_dict[k]['1H'].append(float(data[data['Assignment'] == k]['w2 Hz'].values[0]) + dr_1H # Shift with internal reference to trimer
                       - P9_data[P9_data['Assignment'] == k]['w2 Hz'].values[0]) # Subtract trimer as w,0, so dw = (w,i - w,0) 2021-05-10
               
            P2_dict[k]['13C'].append(float(data[data['Assignment'] == k]['w1 Hz'].values[0]) + dr_13C
                       - P9_data[P9_data['Assignment'] == k]['w1 Hz'].values[0])
            
            # WITH CORRECT REFERENCING ALL INTERNAL REFERENCE PEAKS SHOULD HAVE SAME W1 AND SAME W2!
            print('TRIMER INTERNAL REFERENCE, w2, w1')
            print(P9_data[P9_data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0])
            print(P9_data[P9_data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0])
            print('S210A INTERNAL REFERENCE, w2, w1')
            print(float(data[data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0]) + dr_1H)
            print(float(data[data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0]) + dr_13C)
            
            # Plot internal references for S210A to check referencing lines up with trimer IR
            chax.plot(float(data[data['Assignment'] == 'IRC1-H1']['w2 Hz'].values[0]) + dr_1H,
                      float(data[data['Assignment'] == 'IRC1-H1']['w1 Hz'].values[0]) + dr_13C,'k*',markersize=6)
            
            # Plot peaks with internal referencing correction to check referencing
            chax.plot(P9_data[P9_data['Assignment'] == k]['w2 Hz'].values[0], # Trimer shifts as reference
                      P9_data[P9_data['Assignment'] == k]['w1 Hz'].values[0],'mo',markersize=6)
            
            chax.plot(float(data[data['Assignment'] == k]['w2 Hz'].values[0]) + dr_1H, # S210A DegP
                      float(data[data['Assignment'] == k]['w1 Hz'].values[0]) + dr_13C,'o',markersize=6)
    
    plt.show()    
    
    # Calculate model-free fraction bound
    for k in P2_dict.keys():
        
        P2_dict[k]['FB'] = np.sqrt(np.array(P2_dict[k]['1H'])**2 + np.array(P2_dict[k]['13C'])**2)/P33_fitparams[f"dw_{k.split('-')[0] + k.split('-')[1]}"].values[0]
        print(f"FB for {k} is: {P2_dict[k]['FB']}")
    
    # Fit fraction bound
    #P_dict, C_dict = bindingmodels.make_dictionaries(['3A','6B','9B','12B'])
    #P_dict, C_dict = bindingmodels.make_dictionaries(['3A','6B','9B','12B','15B','18B','21B','24B'])
    #P_dict, C_dict = bindingmodels.make_dictionaries(['3A','6B','9B','12B','15B','18B','21B','24B', '27B', '30B', '33B', '36B'])
    
    state_list = [] # For N-mer sims
    N = 8 # N-mer = 3N
    for x in range(1,N+1):
        if x == 1:
            state_list.append('3A') # Trimer
        else:
            state_list.append(f"{3*x}B") # 3x mer, up to 3N mer
    P_dict, C_dict = bindingmodels.make_dictionaries(state_list)
    
    #sim_func = linearextensionto12mer.linear_extension
    #sim_func_fb = linearextensionto12mer.fractionbound
    #sim_func = linearextensionto24mer.linear_extension
    #sim_func_fb = linearextensionto24mer.fractionbound
    sim_func = linearextensionto24mer_2PDZ1PDZ2.linear_extension
    sim_func_fb = linearextensionto24mer_2PDZ1PDZ2.fractionbound
    #sim_func = linearextensionto36mer.linear_extension
    #sim_func_fb = linearextensionto36mer.fractionbound
    #sim_func = linearextensiontonmer.linear_extension
    #sim_func_fb = linearextensiontonmer.fractionbound
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
    
    ### DO MONTE CARLO ERRORS AND CONFIDENCE BANDS
    MC_iter = 1
    MC_dict, MC_data, errors = MonteCarloErrors(fit_df, result.params, result, MC_iter, fit_constants, sim_func, sim_func_fb, P_dict, C_dict)
    
    print(fit_params)
    print(result.params)
    
    FB_1 = []
    barfig,barax = plt.subplots(1,1)
    for index, concentration in enumerate(P33_fit_LT):
        P_dict, C_dict = sim_func(result.params, fit_constants, 50+273.15, concentration*1e-6, P_dict, C_dict)
        FB_1.append(sim_func_fb(P_dict, fit_constants['model']))
        
        print(f"The population sum at {concentration} uM is {sum(P_dict.values())}")
        
        if index == len(P33_fit_LT)-1:
            for k in P_dict.keys():
                barax.bar(int(k[:-1]),P_dict[k])
            barax.set_xlabel('Oligomer size')
            barax.set_ylabel('Population')
            print('The concentration is:',concentration,'uM')
            plt.show()

    # Upper limit for all 3 bound PDZ
    FB_3 = [1.0 for x in range(len(P33_fit_LT))]
    
    plt.style.use('figure')
    fig, ax = plt.subplots(1,1,figsize=(11,8.64))
    axins = inset_axes(ax,width="40%",height="40%",loc=4,bbox_to_anchor=(0.01,0.05,0.995,0.995), bbox_transform=ax.transAxes)
    colors = ['#74add1','#abdda4','#9970ab','#fdae61','#de77ae']
    colors = colors[::-1]
    ax.plot(P33_fit_LT/1000,P33_mean,'k--',linewidth=4)
    axins.plot(P33_fit_LT/1000,P33_mean,'k--',linewidth=4)
    for index, k in enumerate(P33_exptdict.keys()):
        ax.plot(P33_expt_LT/1000,P33_expt_mean,'kd',markersize=16)
        axins.plot(P33_expt_LT/1000,P33_expt_mean,'kd',markersize=16)
    for index, k in enumerate(P2_dict.keys()):
        ax.plot(np.array(params['Sample']['P2']['Concentrations'])/1000,np.ravel(P2_dict[k]['FB']),'o',color=colors[index],markersize=12)
        axins.plot(np.array(params['Sample']['P2']['Concentrations'])/1000,np.ravel(P2_dict[k]['FB']),'o',color=colors[index],markersize=12)
        
        print('Methyl is:',k,'Color is:',colors[index])
    
    ax.plot(P33_fit_LT/1000,FB_1,'--',color='#ff4bab',linewidth=4)
    ax.plot(P33_fit_LT/1000,FB_3,'--',color='#00edb2',linewidth=4)
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
        FB_calc = sim_func_fb(P_dict, fit_constants['model'])
        
        residuals.append(fit_df['FB'].iloc[i] - FB_calc)
    
    return residuals

### Generate errors by Monte Carlo analysis
def MonteCarloErrors(fit_df, opt_params, fit_result, MC_iter, fit_constants, sim_func, sim_func_fb, P_dict, C_dict):
    
    perfect_data = fit_df.copy() # Make copy of dataframe
    RMSD = np.sqrt(fit_result.chisqr/fit_result.ndata)
    
    for i,v in enumerate(fit_df['FB']): # Overwrite experiment with perfect data
        
        P_dict, C_dict = sim_func(opt_params, fit_constants, fit_df['Temperature'].iloc[i]+273.15, 
                                  fit_df['Concentration'].iloc[i]*1e-6, P_dict, C_dict)
        FB_calc = sim_func_fb(P_dict, fit_constants['model']) 

        perfect_data.loc[(perfect_data.Temperature == perfect_data.Temperature.iloc[i]) &
        (perfect_data.Concentration == perfect_data.Concentration.iloc[i]), 'FB'] = FB_calc
    
    MC_dict = {k:[] for k in opt_params.keys()} # Make dictionary for Monte Carlo parameters
    errors = {k+' error':[] for k in MC_dict.keys()} # Make dictionary for errors
    counter = 1
    for x in range(MC_iter):
        
        print(f"######### The current Monte-Carlo iteration is: {counter} #########")
        
        perturbed_data = perfect_data.copy() # Copy perfect data groups
        
        perturbed_data.FB = perturbed_data.FB + np.random.normal(scale=RMSD, size=np.size(perturbed_data.FB)) # Perturb perfect data for MC analysis
        
        perturbed_result = minimize(objective, opt_params, method='nelder', args=(perturbed_data, 
        P_dict, C_dict, fit_constants, sim_func, sim_func_fb))
        {MC_dict[k].append(perturbed_result.params[k]) for k in perturbed_result.params.keys()}
        
        counter = counter + 1
     
    for k in MC_dict.keys():
        errors[k+' error'].append(np.std(MC_dict[k]))
    
    for k1,k2 in zip(opt_params.keys(),errors.keys()):
        print(f"{k1} = {opt_params[k1].value} +/- {errors[k2][0]}")
        
    with open('MC_parameter_dictionary.pickle','wb') as f:
        pickle.dump(MC_dict,f,protocol=pickle.HIGHEST_PROTOCOL)
    
    # Make simulated curves for plotting confidence intervals
    MC_data = {'Simulated data':[],'Upper bound':[],'Lower bound':[]}
    
    sim_dict = {'Temperature':[50 for x in range(100)],'Concentration':np.linspace(1,4816,100),'FB':np.zeros(100)}
    sim_df = pd.DataFrame(sim_dict)
    for x in range(MC_iter):
            
        MCsimparams = Parameters()
        MCsimparams.add('K1',value=MC_dict['K1'][x].value,vary=True)
        MCsimparams.add('cooperativity',value=1,vary=False)
        
        FB_calc = []
        for i,v in enumerate(sim_df['FB']):
        
            P_dict, C_dict = sim_func(MCsimparams, fit_constants, sim_df['Temperature'].iloc[i]+273.15, 
                                  sim_df['Concentration'].iloc[i]*1e-6, P_dict, C_dict)
            FB_calc.append(sim_func_fb(P_dict, fit_constants['model']))
        
        MC_data['Simulated data'].append(FB_calc) # Store data simulated from Monte Carlo parameters
        
    MC_data['Upper bound'] = np.mean(MC_data['Simulated data'],0) + 1.96*np.std(MC_data['Simulated data'],0) # Calculate upper and lower 95% confidence intervals
    MC_data['Lower bound'] = np.mean(MC_data['Simulated data'],0) - 1.96*np.std(MC_data['Simulated data'],0)
    
    return MC_dict, MC_data, errors

calcfractionbound()
