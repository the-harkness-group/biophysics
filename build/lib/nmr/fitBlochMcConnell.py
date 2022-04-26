#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:25:56 2020

@author: robertharkness
"""

import sys
import numpy as np
import nmrglue as ng
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.linalg import expm
import scipy.fftpack as fftpack
import matplotlib.backends.backend_pdf
from lmfit import minimize, Parameters, fit_report


# Simulating or fitting NMR titration lineshapes using the Bloch-McConnell equations
def main():
    
    # Read data from titration analysis and config file
    expt_params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(expt_params['Fit_data'])
    expt_data = data[data['Assignment'].isin(expt_params['Fit_residues'])]
    
    # Get experimental data from NMR titration files for lineshape fitting
    ls_expt, groups = parse_data(expt_params, data, expt_data)
    
    ### Set up fit parameters
    fit_params = Parameters()
    fit_params.add('Kd',value=expt_params['Kd'],vary=False)
    fit_params.add('kr',value=expt_params['koff'],vary=True)
    fit_params.add('kf',value=fit_params['kr'].value/fit_params['Kd'].value,vary=False)
    
    # Add relaxation, chemical shifts, scaling constants for experimental data
    for ind, group in groups:
        fit_params.add(f"R2a_{ind.split('N-H')[0]}",value=expt_params['Fit_residues'][ind]['R2a'],vary=True)
        fit_params.add(f"R2b_{ind.split('N-H')[0]}",value=expt_params['Fit_residues'][ind]['R2b'],vary=True)
        fit_params.add(f"wa_{ind.split('N-H')[0]}",value=expt_params['Fit_residues'][ind]['wa'],vary=True)
        fit_params.add(f"wb_{ind.split('N-H')[0]}",value=expt_params['Fit_residues'][ind]['wb'],vary=True)
        
        for row_ind, row in group.iterrows():
            fit_params.add(f"c_{ind.split('N-H')[0]}_{row_ind}",value=5e-5,vary=True)

    # Run global fit    
    result = minimize(objective, fit_params, args=(expt_params, expt_data, ls_expt, groups)) # run global fit
    print(fit_report(result)) # print fit report
    
    # Call plotting function
    plot_data(expt_params, result.params, ls_expt, expt_data, groups)


# Extract experimental lineshape data
def parse_data(expt_params, data, expt_data):

    # Extract slices for fitting from experimental data
    ls_expt = {}
    groups = expt_data.groupby('Assignment',sort=False) # Group CSP dataframe according to peak assignment
    for ind, group in groups: # iterate over peaks to get experimental data to fit lineshapes with
        
        ls_expt[ind] = {}
        
        for row_ind, row in group.iterrows(): # iterate over experiments for each peak at different ligand concentrations
            
            ligand = row['Concentration_uM'] # get ligand concentration 
            expt_file = '../sparky/' + row['Experiment_peaklist'].split('.list')[0] + '.ucsf' # get sparky file
            
            nmr_dic,nmr_data = ng.sparky.read(expt_file) # read experiment file
            uc_15n = ng.sparky.make_uc(nmr_dic,nmr_data,0)
            uc_1h = ng.sparky.make_uc(nmr_dic,nmr_data,1)
            
            h1_ppm = uc_1h.ppm_scale() # 1H ppm scale
            n15_ppm = uc_15n.ppm_scale() # 15N ppm scale
             
            h1_shift = row['w2_ppm'] # get 1H chemical shift for peak
            n15_shift = row['w1_ppm'] # get 15N chemical shift for peak
            
            h1_int = nmr_data[uc_15n(str(n15_shift)+"ppm"),:] # get 1H intensity slice at 15N chemical shift for fitting
            n15_int = nmr_data[:,uc_1h(str(h1_shift)+"ppm")] # get 15N intensity slice at 1H chemical shift for fitting
            
            # Write slices to dictionary for fitting            
            ls_expt[ind][f"h1_{ligand}"] = h1_int
            ls_expt[ind][f"n15_{ligand}"] = n15_int
            ls_expt[ind][f"h1ppm_{ligand}"] = h1_ppm
            ls_expt[ind][f"n15ppm_{ligand}"] = n15_ppm
            ls_expt[ind][f"nmr_dic_{ligand}"] = nmr_dic
            ls_expt[ind][f"h1max_{ligand}"] = h1_shift
            ls_expt[ind][f"n15max_{ligand}"] = n15_shift
            
            # Get experimental spectral parameters from NMR data dictionary
            ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['sampling'] = 1./ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['spectral_width']
            ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['sampling'] = 1./ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['spectral_width']
            ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['tacq'] = ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['sampling']*expt_params['w1']['size']
            ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['tacq'] = ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['sampling']*expt_params['w2']['size']
            ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['t_w1'] = np.arange(0,ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['tacq']+ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['sampling'],ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['sampling'])
            ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['t_w2'] = np.arange(0,ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['tacq']+ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['sampling'],ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['sampling'])
            
            # Generate ppm scale using spectrometer frequency and transmitter offset in each dimension
            ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['f_w1'] = np.flip((fftpack.fftshift(fftpack.fftfreq(ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['size'],ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['sampling']))/ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['spectrometer_freq']) + ls_expt[ind][f"nmr_dic_{ligand}"]['w1']['xmtr_freq'])
            ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['f_w2'] = np.flip((fftpack.fftshift(fftpack.fftfreq(ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['size'],ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['sampling']))/ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['spectrometer_freq']) + ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['xmtr_freq'])
                        
        #ls_expt[ind]['lim1'] = ls_expt[ind][f"h1ppm_{ligand}"][(ls_expt[ind][f"h1ppm_{ligand}"] - (ls_expt[ind][f"h1ppm_{ligand}"][group.w2_ppm.argmax()] + 0.1)).argmin()]
        #print(group.w2_ppm.values[np.argmax(group.w2_ppm.values)])
        #print(h1_ppm[np.abs(h1_ppm - group.w2_ppm.values[np.argmax(group.w2_ppm.values)]).argmin()])
        
        # Get fit limits
        ls_expt[ind]['w2_lim1'] = np.ravel(np.where(h1_ppm == h1_ppm[np.abs(h1_ppm - (group.w2_ppm.values[np.argmax(group.w2_ppm.values)] + 0.1)).argmin()]))
        ls_expt[ind]['w2_lim2'] = np.ravel(np.where(h1_ppm == h1_ppm[np.abs(h1_ppm - (group.w2_ppm.values[np.argmin(group.w2_ppm.values)] - 0.1)).argmin()]))
        
        ls_expt[ind]['w1_lim1'] = np.ravel(np.where(n15_ppm == n15_ppm[np.abs(n15_ppm - (group.w1_ppm.values[np.argmax(group.w1_ppm.values)] + 0.3)).argmin()]))
        ls_expt[ind]['w1_lim2'] = np.ravel(np.where(n15_ppm == n15_ppm[np.abs(n15_ppm - (group.w1_ppm.values[np.argmin(group.w1_ppm.values)] - 0.3)).argmin()]))
        
    return ls_expt, groups


# Objective function for fit minimization
def objective(fit_params, expt_params, expt_data, ls_expt, groups):
    
    residual = []
    ls_sim = fid(expt_params, fit_params, expt_data, ls_expt, groups)
    for ind, group in groups:
        
        for row_ind, row in group.iterrows():
            
            ligand = row['Concentration_uM']
            lim1 = ls_expt[ind]['w2_lim1'][0]
            lim2 = ls_expt[ind]['w2_lim2'][0]

            #plt.plot(ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['f_w2'][lim1:lim2],ls_sim[ind][f"Mv_{ligand}"][0][lim1:lim2],'r',label=ligand)
            #plt.plot(ls_expt[ind][f"h1ppm_{ligand}"][lim1:lim2],fit_params[f"c_{ind.split('N-H')[0]}_{row_ind}"]*ls_expt[ind][f"h1_{ligand}"][lim1:lim2],'k')
            #plt.xlim(ls_expt[ind][f"h1ppm_{ligand}"][lim1]+0.1, ls_expt[ind][f"h1ppm_{ligand}"][lim2]-0.1)
            
            residual.append(fit_params[f"c_{ind.split('N-H')[0]}_{row_ind}"]*np.real(ls_expt[ind][f"h1_{ligand}"][lim1:lim2]) - np.real(ls_sim[ind][f"Mv_{ligand}"][0][lim1:lim2]))
        
        #plt.legend()
        #plt.show()
        
    residual = np.concatenate(residual,axis=0)
    print(np.sum(np.square(residual)))
        
    return residual


# Calculate FID
def fid(expt_params, fit_params, expt_data, ls_expt, groups):
    
    # Make lineshape simulation dictionary
    ls_sim = {}
    for ind, group in groups: # iterate over peaks and simulate lineshapes
        
        ls_sim[ind] = {}
        
        for row_ind, row in group.iterrows(): # iterate over experiments for each peak at different ligand concentrations
            
            ligand = row['Concentration_uM']
            
            # Get evolution matrix from populations and Bloch-McConnell equations
            params = thermo_magnetization({'PT':expt_params['PT']*1e-6, 'LT':row['Concentration_uM']*1e-6, 'kf':fit_params['kf'].value, 'kr':fit_params['kr'].value,'model':expt_params['model']})
           
            ls_sim[ind][f"Mt_{ligand}"] = []
            ls_sim[ind][f"Mv_{ligand}"] = []
    
            evolve_params = {f"{k.split('_')[0]}":fit_params[k].value for k in fit_params.keys() if k in (f"R2a_{ind.split('N-H')[0]}", f"R2b_{ind.split('N-H')[0]}", f"wa_{ind.split('N-H')[0]}", f"wb_{ind.split('N-H')[0]}")}  # If P + L <-> PL, kab changes at each [L] so need a subdictionary
            evolve_params['kab'] = params['kab']
            evolve_params['kba'] = params['kba']
            evolve_params['pa'] = params['pa']
            evolve_params['pb'] = params['pb']
            evolve_params['xmtr_freq'] = ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['xmtr_freq']
            evolve_params['spectrometer_freq'] = ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['spectrometer_freq']

            if params['model'] == 'P + L <-> PL':
                
                M0 = np.mat([params['Ma0'], params['Mb0']])
                evolve_params['kab'] = params['kab']
                E = blochmcconnell(evolve_params)
            
            for t in ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['t_w2']:
            
                # Calculate FID according to matrix exponential
                evolve = expm(E * t)
                M = np.dot(evolve,M0.T)
                Mat = M[0,0]
                Mbt = M[1,0]
                ls_sim[ind][f"Mt_{ligand}"].append(np.ravel(Mat + Mbt))
            
            ls_sim[ind][f"Mv_{ligand}"].append(fid_processing(expt_params, ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['spectral_width'], ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['size'], ls_sim[ind][f"Mt_{ligand}"])) # Process simulated FID
            
            #ls_sim[ind][f"Mv_{ligand}"].append(blochmcconnell_analytical(evolve_params, ls_expt[ind][f"h1ppm_{ligand}"]*800.2))
            
    return ls_sim


# Calculate thermodynamic concentrations and initial magnetization
def thermo_magnetization(params):
    
    # Intermolecular association
    if params['model'] == 'P + L <-> PL':
        
        PT = params['PT']
        LT = params['LT']

        Kd = params['kr']/params['kf'] # reverse rate constant/forward
        
        a = (1/Kd)
        b = (1/Kd)*PT - (1/Kd)*LT + 1
        c = -LT
    
        L = (-b + np.sqrt(b**2 - 4.*a*c))/(2.*a)
        
        params['kab'] = np.array(params['kf']*L) # kab is pseudo-first order = kf[L]
        params['kba'] = np.array(params['kr'])
    
    # Update params dictionary with thermodynamic populations and initial magnetization
    params['Ma0'] = params['kba']/(params['kba'] + params['kab'])
    params['Mb0'] = params['kab']/(params['kab'] + params['kba'])
    params['pa'] = params['Ma0']
    params['pb'] = params['Mb0']
    
    return params


# Calculate Bloch-McConnell matrix according to kinetic model
def blochmcconnell(params):
    
    # Unpack parameters
    kab = params['kab']
    kba = params['kba']
    wa = params['wa'] - params['xmtr_freq']*params['spectrometer_freq'] # Shift resonance frequency to be relative to transmitter
    wb = params['wb'] - params['xmtr_freq']*params['spectrometer_freq']
    R2a = params['R2a']
    R2b = params['R2b']

    # Bloch-McConnell two-state exchange
    K = np.mat([[-kab, kba],[kab, -kba]]) # if model is P + L <-> PL, kab is pseudo first-order ie = kf[L]
    R = np.mat([[-R2a, 0],[0, -R2b]])
    F = np.mat([[2.*np.pi*wa*1j, 0],[0, 2.*np.pi*wb*1j]])
    
    # Magnetization evolution matrix is sum of chemical kinetics, relaxation, and frequency matrices
    E = K + R + F
    
    return E


# Two-state ligand binding analytical solution to Bloch-McConnell
def blochmcconnell_analytical(params, v):
    
    print(params)
    v = v*2.*np.pi
    
    # Unpack params
    kab = params['kab']
    kba = params['kba']
    pa = params['pa']
    pb = params['pb']
    wa = params['wa']*2.*np.pi
    wb = params['wb']*2*np.pi
    R2a = params['R2a']
    R2b = params['R2b']
    csim = 50000
    
    dw = wa - wb
    
    tau = ( 1 - pb ) / kba
    
    vb = 0.5 * ( wa + wb )
    
    P = tau * ( R2a * R2b - np.square( vb - v ) + 0.25 * np.square( dw ) ) + pa * R2a + pb * R2b
    
    Q = tau * ( vb - v - 0.5 * ( dw ) * ( pa - pb ) )
    
    U = ( vb - v ) * ( 1 + tau * ( R2a + R2b ) ) + 0.5 * ( dw ) * ( R2b - R2a + pa - pb )
    
    Mv = csim * ( ( P * ( 1 + tau * (pb * R2a + pa * R2b ) ) + Q * U) / ( np.square(P) + np.square (U) ) )
    
    return Mv


# Process simulated FID according to experiment for comparison with experimental data
def fid_processing(expt_params, SW, ZF_size, Mt):
    
    Mt = np.ravel(np.array(Mt))
    
    # Window function
    #Mt_apod = [np.exp(-np.pi*x*expt_params['Window_function']['EM']['lb']/expt_params['SW']) for x in len(time)]
    Mt_apod = ng.process.proc_base.em(Mt, lb=expt_params['Window_function']['EM']['lb']/SW)
    
    # Zero fill
    Mt_apod_zf = ng.process.proc_base.zf_size(Mt_apod, ZF_size, mid=False)
    
    # Fourier transform
    #Mv = ng.process.proc_base.fft(Mt_apod_zf)
    Mv = np.flip(fftpack.fftshift(fftpack.fft(Mt_apod_zf,axis=0)))
    #Mv = np.ravel(fftpack.fftshift(fftpack.fft(Mt_apod_zf,axis=0)))
    
    return Mv


# Plot NMR spectrum
def plot_data(expt_params, opt_params, ls_expt, expt_data, groups):
    
    # Plot parameters
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    fig,ax = plt.subplots(len(expt_params['Fit_residues']),2,figsize=(15,8))

    axis_ind = 0
    color_ind = 0
    colors = ['#fff7f3','#fde0dd','#fcc5c0','#fa9fb5','#f768a1','#dd3497','#ae017e','#7a0177','#49006a']

    ls_sim = fid(expt_params, opt_params, expt_data, ls_expt, groups)

    for ind, group in groups: # iterate over peaks to get experimental data to fit lineshapes with
        
        for row_ind, row in group.iterrows():
            
            ligand = row['Concentration_uM'] # get ligand concentration
            
            lim1 = ls_expt[ind]['w2_lim1'][0]
            lim2 = ls_expt[ind]['w2_lim2'][0]
            
            # Plot data
            #ax[axis_ind][0].plot(ls_expt[ind][f"h1ppm_{ligand}"],ls_expt[ind][f"h1_{ligand}"],color=colors[color_ind])
            ax[axis_ind][1].plot(ls_expt[ind][f"n15ppm_{ligand}"],ls_expt[ind][f"n15_{ligand}"],color=colors[color_ind])
            
            ax[axis_ind][0].plot(ls_expt[ind][f"nmr_dic_{ligand}"]['w2']['f_w2'][lim1:lim2],ls_sim[ind][f"Mv_{ligand}"][0][lim1:lim2],color=colors[color_ind],label=ligand)
            ax[axis_ind][0].plot(ls_expt[ind][f"h1ppm_{ligand}"][lim1:lim2],opt_params[f"c_{ind.split('N-H')[0]}_{row_ind}"]*ls_expt[ind][f"h1_{ligand}"][lim1:lim2],'--',color=colors[color_ind],label=ligand)
            #plt.xlim(ls_expt[ind][f"h1ppm_{ligand}"][lim1]+0.1, ls_expt[ind][f"h1ppm_{ligand}"][lim2]-0.1)
            
            
            if ligand == min(group.Concentration_uM):
                
                h1_ppm_max = np.abs(row['w2_ppm'] - ls_expt[ind][f"h1ppm_{ligand}"]).argmin()
                h1_ymax = ls_expt[ind][f"h1_{ligand}"][h1_ppm_max]
                h1_bot, h1_top = ax[axis_ind][0].get_ylim()
                
                n15_ppm_max = np.abs(row['w1_ppm'] - ls_expt[ind][f"n15ppm_{ligand}"]).argmin()
                n15_ymax = ls_expt[ind][f"n15_{ligand}"][n15_ppm_max]
                n15_bot, n15_top = ax[axis_ind][1].get_ylim()
                
                ax[axis_ind][0].set_xlim(max(group.w2_ppm)+0.05, min(group.w2_ppm)-0.05)
                ax[axis_ind][1].set_xlim(max(group.w1_ppm)+0.25, min(group.w1_ppm)-0.25)
                
                #ax[axis_ind][0].set_ylim(-0.05*h1_ymax, h1_ymax+0.05*h1_ymax)
                ax[axis_ind][1].set_ylim(-0.05*n15_ymax, n15_ymax+0.05*n15_ymax)
                
                ax[axis_ind][0].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
                ax[axis_ind][0].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
                ax[axis_ind][0].yaxis.major.formatter._useMathText = True
                ax[axis_ind][0].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
                
                ax[axis_ind][1].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
                ax[axis_ind][1].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
                ax[axis_ind][1].yaxis.major.formatter._useMathText = True
                ax[axis_ind][1].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
                
                ax[axis_ind][0].set_xlabel('$^{1}H$ ppm',fontsize=14) # Frequency domain plot
                ax[axis_ind][0].set_ylabel('intensity',fontsize=14)
                ax[axis_ind][0].set_title(f"{ind}")
                ax[axis_ind][1].set_xlabel('$^{15}N$ ppm',fontsize=14) # Frequency domain plot
                ax[axis_ind][1].set_ylabel('intensity',fontsize=14)
                ax[axis_ind][1].set_title(f"{ind}")
            
            color_ind = color_ind + 1
            
        axis_ind = axis_ind + 1
        color_ind = 0
     
    plt.tight_layout()
    plt.show()


# Call main function to run program
main()



    