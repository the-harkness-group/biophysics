#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:38:32 2020

@author: robertharkness
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import scipy.fftpack as fftpack
from matplotlib import cm


# Params for simulating or fitting NMR spectra using the Bloch-McConnell equations
# Currently supports two-state intramolecular and intermolecular exchange
def params_setup():
    
    # Model
    #model = 'P + L <-> PL'
    #model = 'F <-> U'
    model = 'P + P <-> P2'
    
    # Kinetics
#    kf = 3.9e7 # PDZ1:2 50C 200 mM NaCl
#    kr = 3e4
    
    kf = 3.9e6
    kr = 3e3
    
#    kf = 3.9e5
#    kr = 3e2
    
#    kf = 3.9e4
#    kr = 3e1
    
#    Kd = 1000e-6
#    kf = 2e6
#    kr = Kd*kf
#    
    # Spin frequencies
    wa = -165 # Hz
    wb = 165 # Hz
    
    # Relaxation
    R2a = 5
    R2b = 15
    
    # Spectral parameters
    SW = 1000 # Hz
    TD = 8192
    sampling = 1/SW
    Tmax = sampling*TD
    time = np.arange(0,Tmax,sampling)
    freq = fftpack.fftshift(fftpack.fftfreq(TD,sampling))
    
    if model == 'P + P <-> P2':
        PT = np.logspace(-6,-2.30103,10)
        LT = np.zeros(len(PT))
    
    if model == 'P + L <-> PL':
        PT = 100e-6
        LT = np.array([0*PT, 0.1*PT, 0.2*PT, 0.4*PT, 0.6*PT, PT, 1.5*PT, 2*PT, 3*PT, 6*PT, 10*PT, 20*PT, 50*PT])
    
    if model == 'F <-> U':
        LT = ['None']
        PT = ['None']
    
    # Parameter dictionary
    params = {'kf':kf, 'kr':kr, 'wa':wa, 'wb': wb, 'R2a':R2a, 'R2b':R2b, 'SW':SW, 
              'TD':TD, 'model':model, 'PT':PT, 'LT':LT, 'Ma0':[], 'Mb0':[], 'pa':[], 
              'pb':[], 'SW':SW, 'TD':TD, 'time':time, 'freq':freq}
    
    # Generate FID through thermo concencentrations and Bloch-McConnell equations, do FT to frequency domain
    Mt, Mv, Mt_2, Mv_2 = fid(params)
    
    # Call plotting function
    plot_data(params, Mt, Mv, Mt_2, Mv_2)
    
    # Print thermodynamic populations
    print(f"\n####### THERMODYNAMIC POPULATIONS FOR MODEL: {params['model']} #######")
    print('pa :',params['pa'])
    print('pb :',params['pb'])
    print('\n')


# Calculate thermodynamic concentrations and initial magnetization
def thermo_magnetization(params):
    
    # Intermolecular homo-association
    if params['model'] == 'P + P <-> P2':
        PT = params['PT']
        
        K = params['kf']/params['kr']
        
        P = (-1 + np.sqrt(1 + 8*K*PT))/(4*K)
        params['kab'] = np.array(2*params['kf']*P)
        params['kba'] = np.array(params['kr'])
    
    # Intermolecular hetero-association
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
    
    # Intramolecular exchange
    if params['model'] == 'F <-> U':
        
        params['kab'] = np.array(params['kf'])
        params['kba'] = np.array(params['kr'])
    
    # Update params dictionary with thermodynamic populations and initial magnetization
    if (params['model'] == 'F <-> U') or (params['model'] == 'P + L <-> PL'):
        params['Ma0'] = params['kba']/(params['kba'] + params['kab'])
        params['Mb0'] = params['kab']/(params['kab'] + params['kba'])
        params['pa'] = params['Ma0']
        params['pb'] = params['Mb0']
        
    if params['model'] == 'P + P <-> P2':
        params['pa'] = P/PT
        params['pb'] = 1 - (P/PT)
        params['Ma0'] = params['pa']
        params['Mb0'] = params['pb']
    
    return params


# Calculate Bloch-McConnell matrix according to kinetic model
def blochmcconnell(params):
    
    # Unpack parameters
    kab = params['kab']
    kba = params['kba']
    wa = params['wa']
    wb = params['wb']
    R2a = params['R2a']
    R2b = params['R2b']

    # Bloch-McConnell two-state exchange
    # If model is P + L <-> PL, kab is pseudo first-order ie = kf*[L]
    # If model is P + P <-> P2, kab is pseudo first-order 2*kf*[P]
    K = np.mat([[-kab, kba],[kab, -kba]])
    R = np.mat([[-R2a, 0],[0, -R2b]])
    F = np.mat([[2.*np.pi*wa*1j, 0],[0, 2.*np.pi*wb*1j]])
    
    # Magnetization evolution matrix is sum of chemical kinetics, relaxation, and frequency matrices
    E = K + R + F
    
    return E
   

# Calculate FID
def fid(params):

    # Get evolution matrix from populations and Bloch-McConnell equations
    params = thermo_magnetization(params)

    Mt = {f"Mt_{x}":[] for x in range(len(params['LT']))}
    Mv = {f"Mv_{x}":[] for x in range(len(params['LT']))}
    Mt_2 = {f"Mt_{x}":[] for x in range(len(params['LT']))}
    Mv_2 = {f"Mv_{x}":[] for x in range(len(params['LT']))}
    
    evolve_params = {k: params[k] for k in ('R2a', 'R2b', 'wa', 'wb', 'kba')} # If P + L <-> PL, kab changes at each [L] so need a subdictionary

    # If model is P + L <-> PL, need to loop over FIDs for each ligand concentration
    # In simple F <-> U model, this will revert to one FID calculation since LT = 0
    for num, ligand in enumerate(params['LT']):
        
        if params['model'] == 'P + L <-> PL':
            M0 = np.mat([params['Ma0'][num], params['Mb0'][num]])
            evolve_params['kab'] = params['kab'][num]
            E = blochmcconnell(evolve_params)
            
        if params['model'] == 'F <-> U': # LT is None
            M0 = np.mat([params['Ma0'], params['Mb0']])
            evolve_params['kab'] = params['kab']            
            E = blochmcconnell(evolve_params)
            
        if params['model'] == 'P + P <-> P2': # LT is filled with None
            M0 = np.mat([params['Ma0'][num], params['Mb0'][num]])
            evolve_params['kab'] = params['kab'][num]            
            E = blochmcconnell(evolve_params)
            
        for t in params['time']:
            
            # Calculate FID according to matrix exponential
            evolve = expm(E * t)
            M = np.dot(evolve,M0.T)
            Mat = M[0,0]
            Mbt = M[1,0]
            #print(f"The ratio of M[0,0]/M[1,0] is: {np.real(M[0,0])/np.real(M[1,0])}")
            Mt[f"Mt_{num}"].append(np.ravel(Mat + Mbt))
            Mt_2[f"Mt_{num}"].append(np.ravel(Mat))
            #Mt[f"Mt_{num}"].append(np.ravel(Mat))
#            if np.abs(np.real(Mat)) > np.abs(np.real(Mbt)):
#                Mt[f"Mt_{num}"].append(np.ravel(Mat))
#                print('Mat')
#            #Mt[f"Mt_{num}"].append(np.ravel(Mbt))
#            if np.abs(np.real(Mat)) < np.abs(np.real(Mbt)):
#                Mt[f"Mt_{num}"].append(np.ravel(Mbt))
#                print('Mbt')
        
        Mv[f"Mv_{num}"].append(np.ravel(fftpack.fftshift(fftpack.fft(Mt[f"Mt_{num}"],axis=0))))
        Mv_2[f"Mv_{num}"].append(np.ravel(fftpack.fftshift(fftpack.fft(Mt_2[f"Mt_{num}"],axis=0))))
    
    return Mt, Mv, Mt_2, Mv_2


# Plot NMR spectrum
def plot_data(params, Mt, Mv, Mt_2, Mv_2):
    
    # Plot parameters
    plt.style.use('figure')
    label_params = {'mathtext.default': 'regular' }   
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    cm_space = np.linspace(0,1,len(params['LT']))
    colors = [cm.turbo(x) for x in cm_space]
    fig, ax = plt.subplots(1,2,figsize=(20,7.273)) # Plot for overlaying final results
    paperfig, paperax = plt.subplots(1,1)
    ax[0].set_xlabel('time [s]',fontsize=14) # Time domain plot
    ax[0].set_ylabel('intensity',fontsize=14)
    ax[1].set_xlabel('frequency [Hz]',fontsize=14) # Frequency domain plot
    ax[1].set_ylabel('intensity',fontsize=14)
    ax[1].set_xlim([params['wa']-100,params['wb']+100])
    paperax.set_xlabel('Frequency Hz')
    paperax.set_ylabel('Intensity')
    paperax.set_xlim([params['wa']-50,params['wb']+50])
    
    fig2,ax2 = plt.subplots(1,1)
    #ax2.set_xlabel('Concentration uM')
    #ax2.set_ylabel('Fraction bound')
    ax2.set_xlabel('Simulated $P_{6}$')
    ax2.set_ylabel('Apparent $P_{6}$')
    #ax2.set_ylim([-0.01,1.01])
    ax2.plot(params['pb'],params['pb'],'k--',linewidth=2)
    ax2.set_xlim([-0.02,max(params['pb']+0.02)])
    ax2.set_ylim([-0.02,max(params['pb']+0.02)])
    
    for num, axis in enumerate(ax):
        ax[num].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        ax[num].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        ax[num].yaxis.major.formatter._useMathText = True
        ax[num].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
    
    for num, ligand in enumerate(params['LT']):
        
        if params['model'] == 'P + L <-> PL':
            
            # Time domain
            ax[0].plot(params['time'], Mt[f"Mt_{num}"],color=colors[num])
            
            # Frequency domain
            ax[1].plot(params['freq'], np.real(Mv[f"Mv_{num}"][0][:]),color=colors[num],label=f"$L_{{T}}$ = {ligand*1e6} $\mu$M")
            
            csp_max = np.argmax(np.real(Mv[f"Mv_{num}"][0][:]))
            csp = params['freq'][csp_max] - params['wa']
            dw = params['wb'] - params['wa']
            #print('Apparent fraction bound at LT',ligand*1e6,'is:',csp/dw)
            #print('Frequency at peak max is',params['freq'][csp_max])
            ax2.plot(ligand*1e6,csp/dw,'ko')
            
        if params['model'] == 'P + P <-> P2':
            
            # Time domain
            ax[0].plot(params['time'], Mt[f"Mt_{num}"],color=colors[num])
            #ax[0].plot(params['time'], Mt[f"Mt_{num}"],color=colors[num])
            
            # Frequency domain
            ax[1].plot(params['freq'], np.real(Mv[f"Mv_{num}"][0][:]),color=colors[num],label=f"$P_{{T}}$ = {np.round(params['PT'][num]*1e6,1)} $\mu$M")
            #ax[1].plot(params['freq'], np.real(Mv_2[f"Mv_{num}"][0][:]),'--',color=colors[num],label=f"$P_{{T}}$ = {np.round(params['PT'][num]*1e6,1)} $\mu$M")
            
            paperax.plot(params['freq'], np.real(Mv[f"Mv_{num}"][0][:]),color=colors[num],
                         label=f"$P_{{T}}$ = {np.round(params['PT'][num]*1e6,1)} $\mu$M")
            
            print(f"kex = {params['kab'][num] + params['kba']} /s")
            
            csp_max = np.argmax(np.real(Mv[f"Mv_{num}"][0][:]))
            csp = params['freq'][csp_max] - params['wa']
            dw = params['wb'] - params['wa']
            #print('Apparent fraction bound at LT',ligand*1e6,'is:',csp/dw)
            #print('Frequency at peak max is',params['freq'][csp_max])
            #ax2.plot(params['PT'][num]*1e6,csp/dw,'rd',label='Apparent')
            #ax2.plot(params['PT'][num]*1e6,params['pb'][num],'ko',label='Real')
            ax2.plot(params['pb'][num],csp/dw,'o',markersize=8,markerfacecolor=colors[num],markeredgecolor=colors[num])
            
        if params['model'] == 'F <-> U':
            
            # Time domain
            ax[0].plot(params['time'], Mt[f"Mt_{num}"],color='k')
            
            # Frequency domain
            ax[1].plot(params['freq'], np.real(Mv[f"Mv_{num}"][0][:]),color='k')
        
    ax[1].legend(loc='upper right',frameon=False)
    #ax2.legend(loc='upper right',frameon=False)
    plt.tight_layout()
    #paperax.legend(loc='upper right',frameon=False)
    paperfig.tight_layout()
    paperfig.savefig('BlochMcConnell_1.pdf',format='pdf')
    fig2.tight_layout()
    fig2.savefig('Apparent_FB_1.pdf',format='pdf')
    #plt.show()

# Call main function to run program
params_setup()
    
    
