#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:07:10 2021

@author: robertharkness
"""

import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, report_fit
import pandas as pd
import branched36
import customcolormap
import bindingmodels
import hydrodynamics
import matplotlib.backends.backend_pdf
import time

# Initial function
def main():
    
    params = yaml.safe_load(open(sys.argv[1],'r'))        
    DLS = pd.read_csv(params['DLS']) # Get DLS data for global fit
    DLS = DLS[(DLS['Sample'] == 'P2') & (DLS['Concentration'] <= 200)] # DegP2
    DLS.Temperature = DLS.Temperature + 273.15 # Convert T to K
    DLS.Concentration = DLS.Concentration*1e-6 # Convert CT to M
    NMR = pd.read_csv(params['NMR']) # Get NMR data for global fit
    NMR.Temperature = NMR.Temperature + 273.15 # Convert T to K
    NMR.Concentration = NMR.Concentration*1e-6 # Convert CT to M
    
    R = bindingmodels.constants()
    To = 25 + 273.15
    #dH_NMRo = -15.2 # PDZ1:2 from NMR titration U13C15N1H, 200 mM NaCl, 25 and 50C
    #K_NMRo = 1/444e-6
    #K_NMR = 1/713e-6
    #dCp_NMR = -0.57
    K_NMR = 1/376e-6 # From PDZ1:2 ILVM 50C 1M NaCl D2O, assuming same dH as above
    K_NMRo = 1/234e-6 # 25C
    dCp_NMR = -0.57 # From DLS fit
    Temperatures = NMR.Temperature.iloc[0] # Calculate dH for PDZ1:2
    dH_NMRo = (np.log(K_NMR/K_NMRo)*R - dCp_NMR*(np.log(Temperatures/To) + (To/Temperatures) - 1))/((1/To) - (1/Temperatures))

    #dH1o = -136.46 # 3-6A 200 mM NaCl
    #K1o = 707673.630 # Without explicit 9/1 scaling
    #K1o = 78630.4 # With explicit 9/1 scaling
    #dCp1 = -6.743
    
    dH1o = -99.14 # 3-6A 1M NaCl
    K1o = 70266.13
    dCp1 = -7.41

    # Hydrodynamic constants
    Rh = 4.89e-9 # M3
    #eta_coeffs = [-6.32053e-9, 6.08559e-06, -1.96431e-3, 2.13120e-1] # 200 mM NaCl
    eta_coeffs = [-6.83066e-9, 6.57839e-6, -2.12377e-3, 2.30446e-1] # 1M NaCl
    
    fit_params = Parameters() # Set up dictionary of fit parameters
    fit_params.add('K1o',value=76399.8,vary=False)
    fit_params.add('dH1o',value=-76.2,vary=False)
    fit_params.add('dCp1',value=-6.63,vary=False)
    fit_params.add('K2o',value=K_NMRo,vary=False)
    fit_params.add('dH2o',value=dH_NMRo,vary=False)
    fit_params.add('dCp2',value=dCp_NMR,vary=False)
    fit_params.add('alpha',value=0.9565,vary=True)
    fit_params.add('beta',value=2.8053,vary=True)
    fit_params.add('gamma',value=20,vary=True)
#    fit_params.add('K3o',value=10*K_NMRo,vary=True,min=0)
#    fit_params.add('dH3o',value=dH_NMRo,vary=True)
#    fit_params.add('dCp3',value=dCp_NMR,vary=False)
#    fit_params.add('K4o',value=15*K_NMRo,vary=True,min=0)
#    fit_params.add('dH4o',value=dH_NMRo,vary=True)
#    fit_params.add('dCp4',value=dCp_NMR,vary=False)
    
    P_dict, D_dict, C_dict = branched36.makedictionaries() # Get dictionaries
    simfunc = branched36.branchedto36 # Define function to simulate data
#    result = minimize(objective, fit_params, method='leastsq', args=(P_dict, 
#            D_dict, C_dict, DLS, NMR, To, Rh, eta_coeffs, simfunc)) # Do fit
#    
#    plotting(result.params, DLS, NMR, To, P_dict, D_dict, 
#             C_dict, Rh, eta_coeffs, simfunc) # Plot fit results
#   
#    try:
#        report_fit(result)
#    except:
#        try:
#            print('Fit report broke')
#            print(result.params)
#        except:
#            print('You are plotting, no result params')
    
    plotting(fit_params, DLS, NMR, To, P_dict, D_dict, C_dict, 
             Rh, eta_coeffs, simfunc)
    
# Minimization function
def objective(fit_params, P_dict, D_dict, C_dict, DLS, NMR, To,
              Rh, eta_coeffs, simfunc):
    
#    fig, axs = plt.subplots(1,2,figsize=(13,5.7))
#    axs[1].set_ylim(1.0e-7,9e-7)
#    axs[1].set_xlabel('Temperature C')
#    axs[1].set_ylabel('Dz cm^2 s^-1')
#    axs[0].set_ylim([-0.01,1.01])
#    axs[0].set_xlabel('CT uM')
#    axs[0].set_ylabel('Fraction bound')
    
    rss1 = [] # Scale rss by data unit size and by num points
    for y, temperature in enumerate(NMR.Temperature): # NMR
        #time1 = time.time()
        P_dict, D_dict, C_dict = simfunc(fit_params, P_dict, D_dict, C_dict, 
                NMR.Concentration[y], NMR.Temperature[y], To, Rh, eta_coeffs)
        #time2 = time.time()
        #print(f"One interation takes: {time2-time1}s")
        rss1.append((NMR['Fraction bound'][y] - P_dict['FB'])
                       /NMR['Fraction bound'][y])
##            
#        rss1.append((NMR['Fraction bound'][y] - P_dict['FB']))
#            
#        #axs[0].plot(NMR.Concentration[y]*1e6,NMR['Fraction bound'][y],'ko')
#        #axs[0].plot(NMR.Concentration[y]*1e6,P_dict['FB'],'ro')
    rss1 = np.array(rss1)/len(NMR.Temperature)
    
    rss2 = []
    for y, temperature in enumerate(DLS.Temperature): # DLS
        P_dict, D_dict, C_dict = simfunc(fit_params, P_dict, D_dict, C_dict, 
                DLS.Concentration.iloc[y], DLS.Temperature.iloc[y], To, Rh, eta_coeffs)
        rss2.append((DLS.D.iloc[y]*1e4 - D_dict['Dz']*1e4)
                        /(DLS.D.iloc[y]*1e4))
            
#        rss2.append(DLS.D.iloc[y]*1e4 - D_dict['Dz']*1e4)
            
#        axs[1].plot(DLS.Temperature[y]-273.15,DLS['D'][y]*1e4,'ko')
#        axs[1].plot(DLS.Temperature[y]-273.15,D_dict['Dz']*1e4,'ro')
#        axs[1].plot(DLS.Temperature[y]-273.15,D_dict['3A']*1e4,'go')
#        axs[1].plot(DLS.Temperature[y]-273.15,D_dict['24E']*1e4,'co')
#        axs[1].plot(DLS.Temperature[y]-273.15,D_dict['36G']*1e4,'bo')
    rss2 = np.array(rss2)/len(DLS.Temperature)
    
    #plt.ion()
    #plt.pause(0.00001)
    #plt.ioff()
    #plt.close()
    
    rss = np.concatenate((rss1,rss2))
    #rss = rss1
    #rss = rss2
        
    return rss

# Plot simulations/fit
def plotting(opt_params, DLS, NMR, To, P_dict, D_dict, C_dict, Rh, eta_coeffs, simfunc):
    
    plt.style.use('figure')
    hex_colors = ['#020101','#281112','#4E2128','#733243','#984364','#BD548A','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF']
    hex_colors = list(reversed(hex_colors))
    M3_color = '#fc8d59'
    M6_color = '#78c679'
    M24_color = '#2b8cbe'
    M36_color = '#5e4fa2'
    pdf = matplotlib.backends.backend_pdf.PdfPages('DegPsimulations.pdf')
    
    sim_Temperature = np.array([10,20,30,40,50]) + 273.15 # Make simulation dictionary
    sim_Concentration = np.array([10, 25, 50, 100, 200, 500, 1000])*1e-6
    sim_dict = {'Temperature':[],'Concentration':[]}
    for x in sim_Concentration:
        for y in sim_Temperature:
            sim_dict['Temperature'].append(y)
            sim_dict['Concentration'].append(x)
    sim_df = pd.DataFrame(sim_dict)
        
    for y, temperature in enumerate(sim_df.Temperature): # Population histograms

            P_dict, D_dict, C_dict = simfunc(opt_params, P_dict, D_dict, C_dict, 
            sim_df.Concentration[y], sim_df.Temperature[y], To, Rh, eta_coeffs)
    
            fig, ax = plt.subplots(1,1,figsize=(12,5))
            ax.bar(3,P_dict['3A'],width=0.5)
            ax.bar(5.5,P_dict['6A'],width=0.5)
            ax.bar(6,P_dict['6B'],width=0.5)
            ax.bar(8.5,P_dict['9B'],width=0.5)
            ax.bar(9,P_dict['9C'],width=0.5)
            ax.bar(11,P_dict['12B'],width=0.5)
            ax.bar(11.5,P_dict['12C'],width=0.5)
            ax.bar(12,P_dict['12D'],width=0.5)
            ax.bar(12.5,P_dict['12E'],width=0.5)
            ax.bar(14.0,P_dict['15B'],width=0.5)
            ax.bar(14.5,P_dict['15D'],width=0.5)
            ax.bar(15.0,P_dict['15E'],width=0.5)
            ax.bar(15.5,P_dict['15F'],width=0.5)
            ax.bar(17,P_dict['18D'],width=0.5)
            ax.bar(17.5,P_dict['18E'],width=0.5)
            ax.bar(18,P_dict['18F'],width=0.5)
            ax.bar(18.5,P_dict['18G'],width=0.5)
            ax.bar(20.5,P_dict['21E'],width=0.5)
            ax.bar(21,P_dict['21F'],width=0.5)
            ax.bar(21.5,P_dict['21G'],width=0.5)
            ax.bar(23.5,P_dict['24E'],width=0.5)
            ax.bar(24,P_dict['24F'],width=0.5)
            ax.bar(24.5,P_dict['24G'],width=0.5)
            ax.bar(26.5,P_dict['27F'],width=0.5)
            ax.bar(27,P_dict['27G'],width=0.5)
            ax.bar(29.5,P_dict['30F'],width=0.5)
            ax.bar(30,P_dict['30G'],width=0.5)
            ax.bar(33,P_dict['33G'],width=0.5)
            ax.bar(36,P_dict['36G'],width=0.5)
            ax.set_ylim([-0.01,1.01])
            ax.set_xticks([3,6,9,12,15,18,21,24,27,30,33,36])
            ax.set_xlabel('Oligomer size')
            ax.set_ylabel('Population')
            ax.set_title("%s \N{DEGREE SIGN}C, %s $\mu$M"
                         %(sim_df.Temperature[y]-273.15,
                           np.round(sim_df.Concentration[y]*1e6)))
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    
    # NMR titration at 50C
    temperature = 50 + 273.15
    # For comparison with PDZ1:2 titration
    # Start of Path B, 2 PDZ bound
    CT_NMR = np.logspace(-6,-2.3010299956639813,100)
    K2 = bindingmodels.equilibriumconstants(opt_params['K2o'].value,
        opt_params['dH2o'].value, opt_params['dCp2'].value,To, temperature)
    PB_NMR = bindingmodels.onesite(CT_NMR, 100e-6, K2)
    
    best_fit_NMR = np.zeros(len(CT_NMR)) # Get best fit data
    pop_sum = np.zeros(len(CT_NMR))
    for x, concentration in enumerate(CT_NMR):
        P_dict, D_dict, C_dict = simfunc(opt_params, P_dict, D_dict, C_dict,
        concentration, temperature, To, Rh, eta_coeffs)
        best_fit_NMR[x] =  P_dict['FB']
        pop_sum[x] = P_dict['Psum']
    
    fig, ax = plt.subplots(1,1)
    ax.set_ylim([-0.01,1.01])
    ax.set_xticks([0,1000,2000,3000,4000,5000])
    ax.set_xlabel('$C_{T}$ $\mu$M')
    ax.set_ylabel('Population')
    ax.set_title("%s \N{DEGREE SIGN}C"%(temperature-273.15))
    L272_A = NMR[NMR['Assignment'] == 'L272_A']
    L272_B = NMR[NMR['Assignment'] == 'L272_B']
    ax.plot(L272_A.Concentration*1e6,L272_A['Fraction bound'],'ko',label='L272A')
    ax.plot(L272_B.Concentration*1e6,L272_B['Fraction bound'],'o',color='#66c2a5',label='L272B')
    ax.plot(CT_NMR*1e6,best_fit_NMR,'--',color='#d53e4f',linewidth=2,label='Fit')
    ax.plot(CT_NMR*1e6,PB_NMR,'--',color='#3288bd',linewidth=2,label='PDZ1:2')
    ax.plot(CT_NMR*1e6,pop_sum,'k--',linewidth=2,label='Population sum')
    ax.legend(frameon=False,prop={'size':18},loc='lower right')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    
    # DLS data
    fig2, ax2 = plt.subplots(1,1)
    ax2.set_xlabel('Temperature \N{DEGREE SIGN}C')
    ax2.set_ylabel('$D_{z}$ cm$^{2}$ s$^{-1}$')
    DLS_groups = DLS.groupby('Concentration')
    cmap = customcolormap.get_continuous_cmap(hex_colors,len(DLS_groups))
    for x, (ind, group) in enumerate(DLS_groups):
        Dz = np.zeros(len(group))
        for y, temperature in enumerate(group.Temperature):
            
            P_dict, D_dict, C_dict = simfunc(opt_params, P_dict, D_dict, C_dict,
            group.Concentration.iloc[y], group.Temperature.iloc[y], To, Rh, eta_coeffs)
            
            Dz[y] = D_dict['Dz']
        
        ax2.plot(group.Temperature-273.15,group.D*1e4,'o',color=cmap[x])
        ax2.plot(group.Temperature-273.15,Dz*1e4,linewidth=2,color=cmap[x])
    
    DLS_Temperature = np.linspace(5,50,19) + 273.15
    D3 = np.zeros(len(DLS_Temperature))
    D6 = np.zeros(len(DLS_Temperature))
    D24 = np.zeros(len(DLS_Temperature))
    D36 = np.zeros(len(DLS_Temperature))
    for y, temperature in enumerate(DLS_Temperature):
        
        eta = hydrodynamics.viscosity(temperature, eta_coeffs)
        D3[y] = hydrodynamics.stokes_diffusion(temperature, eta, Rh)
        D6[y] = hydrodynamics.scaled_diffusion(D3[y],2,-0.227)
        D24[y] = hydrodynamics.scaled_diffusion(D3[y],8,-0.333)
        D36[y] = hydrodynamics.scaled_diffusion(D3[y],12,-0.333)
        
    ax2.plot(DLS_Temperature-273.15,D3*1e4,'--',linewidth=2,color=M3_color)
    ax2.plot(DLS_Temperature-273.15,D6*1e4,'--',linewidth=2,color=M6_color)
    ax2.plot(DLS_Temperature-273.15,D24*1e4,'--',linewidth=2,color=M24_color)
    ax2.plot(DLS_Temperature-273.15,D36*1e4,'--',linewidth=2,color=M36_color)
    
    fig2.tight_layout(pad=0.4)
    pdf.savefig(fig2)
    pdf.close()
    
main()