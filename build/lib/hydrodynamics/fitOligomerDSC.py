#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:50:38 2019

@author: robertharkness
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
from lmfit import minimize, Parameters, report_fit
import pickle

### Model for monomer-trimer-hexamer-dodecamer equilibrium
def MTHD(fit_params, Temperatures, CT0, Cp_U):
    
    # Set-up concentrations, parameters for calculating DSC Cp profiles
    CT1 = np.copy(CT0)
    CT_eq = np.copy(CT0)
    dH_M0 = fit_params['dH_M'].value
    dS_M0 = fit_params['dS_M'].value
    dH_Tr0 = fit_params['dH_Tr'].value
    dS_Tr0 = fit_params['dS_Tr'].value
    dH_H0 = fit_params['dH_H'].value
    dS_H0 = fit_params['dS_H'].value
    dH_Dd0 = fit_params['dH_Dd'].value
    dS_Dd0 = fit_params['dS_Dd'].value
    dH_agg0 = fit_params['dH_agg'].value
    kagg0 = fit_params['kagg0'].value
    Eagg = fit_params['Eagg'].value
    SR = fit_params['SR'].value
    T_inc = fit_params['T_inc'].value
    
    dCp_M = fit_params['dCp_M'].value
    dCp_Tr = fit_params['dCp_Tr'].value
    dCp_H = fit_params['dCp_H'].value
    dCp_Dd = fit_params['dCp_Dd'].value

    U = np.zeros([len(Temperatures),len(CT1)])
    M = np.zeros([len(Temperatures),len(CT1)])
    Tr = np.zeros([len(Temperatures),len(CT1)])
    H = np.zeros([len(Temperatures),len(CT1)])
    Dd = np.zeros([len(Temperatures),len(CT1)])
    Agg = np.zeros([len(Temperatures),len(CT1)])

    P_U = np.zeros([len(Temperatures),len(CT1)])
    P_M = np.zeros([len(Temperatures),len(CT1)])
    P_Tr = np.zeros([len(Temperatures),len(CT1)])
    P_H = np.zeros([len(Temperatures),len(CT1)])
    P_Dd = np.zeros([len(Temperatures),len(CT1)])
    P_agg = np.zeros([len(Temperatures),len(CT1)])
    P_sum = np.zeros([len(Temperatures),len(CT1)])
    
    dPUdT = np.zeros([len(Temperatures),len(CT1)])
    dPMdT = np.zeros([len(Temperatures),len(CT1)])
    dPTrdT = np.zeros([len(Temperatures),len(CT1)])
    dPHdT = np.zeros([len(Temperatures),len(CT1)])
    dPDddT = np.zeros([len(Temperatures),len(CT1)])
    dPaggdT = np.zeros([len(Temperatures),len(CT1)])
    
    #dCp_M = np.zeros([len(Temperatures),len(CT1)])
    #dCp_Tr = np.zeros([len(Temperatures),len(CT1)])
    #dCp_H = np.zeros([len(Temperatures),len(CT1)])
    #dCp_Dd = np.zeros([len(Temperatures),len(CT1)])
    
    dH_M = np.zeros([len(Temperatures),len(CT1)])
    dH_Tr = np.zeros([len(Temperatures),len(CT1)])
    dH_H = np.zeros([len(Temperatures),len(CT1)])
    dH_Dd = np.zeros([len(Temperatures),len(CT1)])
    
    dS_M = np.zeros([len(Temperatures),len(CT1)])
    dS_Tr = np.zeros([len(Temperatures),len(CT1)])
    dS_H = np.zeros([len(Temperatures),len(CT1)])
    dS_Dd = np.zeros([len(Temperatures),len(CT1)])
    
    dG_M = np.zeros([len(Temperatures),len(CT1)])
    dG_Tr = np.zeros([len(Temperatures),len(CT1)])
    dG_H = np.zeros([len(Temperatures),len(CT1)])
    dG_Dd = np.zeros([len(Temperatures),len(CT1)])
    
    K_M = np.zeros([len(Temperatures),len(CT1)])
    K_Tr = np.zeros([len(Temperatures),len(CT1)])
    K_H = np.zeros([len(Temperatures),len(CT1)])
    K_Dd = np.zeros([len(Temperatures),len(CT1)])
    k_agg = np.zeros([len(Temperatures),len(CT1)])
    
    #Cp_U = fit_params['m'].value*Temperatures + fit_params['b'].value
    
    Cp_MTHD = np.zeros([len(Temperatures),len(CT1)])
    
    #dH_tot = dH_Tr0 + dH_Dd0
    #dCp_M = (Cp_F - Cp_U)*(dH_M0/dH_tot)
    #dCp_Tr = (Cp_F - Cp_U)*(dH_Tr0/dH_tot)
    #dCp_H = (Cp_F - Cp_U)*(dH_H0/dH_tot)
    #dCp_Dd = (Cp_F - Cp_U)*(dH_Dd0/dH_tot)
    
    # Loop through temperature and concentration data to calculate Cp
    for y in range(len(Temperatures)):
        
        for x in range(len(CT1)):
        
            # Calculate equilibrium constants
            #dH_M[y,x] = dH_M0 + (Temperatures[y] - Temperatures[0])*dCp_M[y]             # Temperature-dependent dCps
            #dS_M[y,x] = dS_M0 + np.log(Temperatures[y]/Temperatures[0])*dCp_M[y]
            #dH_Tr[y,x] = dH_Tr0 + (Temperatures[y] - Temperatures[0])*dCp_Tr[y]
            #dS_Tr[y,x] = dS_Tr0 + np.log(Temperatures[y]/Temperatures[0])*dCp_Tr[y]
            #dH_H[y,x] = dH_H0 + (Temperatures[y] - Temperatures[0])*dCp_H[y]
            #dS_H[y,x] = dS_H0 + np.log(Temperatures[y]/Temperatures[0])*dCp_H[y]
            #dH_Dd[y,x] = dH_Dd0 + (Temperatures[y] - Temperatures[0])*dCp_Dd[y]
            #dS_Dd[y,x] = dS_Dd0 + np.log(Temperatures[y]/Temperatures[0])*dCp_Dd[y]
            
            dH_M[y,x] = dH_M0 + (Temperatures[y] - Temperatures[0])*dCp_M             # Temperature-independent dCps
            dS_M[y,x] = dS_M0 + np.log(Temperatures[y]/Temperatures[0])*dCp_M
            dH_Tr[y,x] = dH_Tr0 + (Temperatures[y] - Temperatures[0])*dCp_Tr
            dS_Tr[y,x] = dS_Tr0 + np.log(Temperatures[y]/Temperatures[0])*dCp_Tr
            dH_H[y,x] = dH_H0 + (Temperatures[y] - Temperatures[0])*dCp_H
            dS_H[y,x] = dS_H0 + np.log(Temperatures[y]/Temperatures[0])*dCp_H
            dH_Dd[y,x] = dH_Dd0 + (Temperatures[y] - Temperatures[0])*dCp_Dd
            dS_Dd[y,x] = dS_Dd0 + np.log(Temperatures[y]/Temperatures[0])*dCp_Dd
            
            dG_M[y,x] = dH_M[y,x] - Temperatures[y]*dS_M[y,x]
            dG_Tr[y,x] = dH_Tr[y,x] - Temperatures[y]*dS_Tr[y,x]
            dG_H[y,x] = dH_H[y,x] - Temperatures[y]*dS_H[y,x]
            dG_Dd[y,x] = dH_Dd[y,x] - Temperatures[y]*dS_Dd[y,x]
        
            K_M[y,x] = np.exp(-dG_M[y,x]/(R*Temperatures[y]))
            K_Tr[y,x] = np.exp(-dG_Tr[y,x]/(R*Temperatures[y]))
            K_H[y,x] = np.exp(-dG_H[y,x]/(R*Temperatures[y]))
            K_Dd[y,x] = np.exp(-dG_Dd[y,x]/(R*Temperatures[y]))
            k_agg[y,x] = kagg0*np.exp((Eagg/R)*((1/Temperatures[0]) - (1/Temperatures[y])))
            
            # Get free monomer concentration from real, positive root of binding polynomial
            # Used to calculate concentrations of other oligomers
            rr = np.roots([12*K_Dd[y,x]*np.square(K_H[y,x])*np.power(K_Tr[y,x],4)*np.power(K_M[y,x],12), 0, 0, 0, 0, 0, 6*K_H[y,x]*np.square(K_Tr[y,x])*np.power(K_M[y,x],6), 0, 0, 3*K_Tr[y,x]*np.power(K_M[y,x],3), 0, (K_M[y,x] + 1), -CT_eq[x]])
            
            for root in rr:
                if np.isreal(root) and np.real(root) > 0:
                    
                    U[y,x] = np.real(root)
                    M[y,x] = K_M[y,x]*U[y,x]
                    Tr[y,x] = K_Tr[y,x]*np.power(M[y,x],3)
                    H[y,x] = K_H[y,x]*np.square(Tr[y,x])
                    Dd[y,x] = K_Dd[y,x]*np.square(H[y,x])
                    
                    # Aggregation kinetics - first point is just increase in aggregted state from the unfolded state
                    # Assumes aggregation is first order kinetic process
                    # The amount of aggregation is calculated as total unfolded minus how much unfolded remains after aggregation for a time t
                    if y == 0:
                        Agg[y,x] = U[y,x] - U[y,x]*np.exp(-k_agg[y,x]*T_inc/SR)
                    
                    # Aggregation kinetics - point y is the sum of y-1 and the increase in aggregated state from the unfolded state
                    # Assumes aggregation is first order kinetic process
                    else:
                        Agg[y,x] = Agg[y-1,x] + (U[y,x] - U[y,x]*np.exp(-k_agg[y,x]*T_inc/SR))
            
            # Aggregation kinetics - the total monomer concentration available
            # for equilibration between states gets knocked down every point
            # due to aggregation from the unfolded state
            CT_eq[x] = CT_eq[x] - (U[y,x] - U[y,x]*np.exp(-k_agg[y,x]*T_inc/SR))
            
            # Populations of each state are calculated using the total concentration, not just the equilibration CT
            P_U[y,x] = U[y,x]/CT1[x]
            P_M[y,x] = M[y,x]/CT1[x]
            P_Tr[y,x] = 3*Tr[y,x]/CT1[x]
            P_H[y,x] = 6*H[y,x]/CT1[x]
            P_Dd[y,x] = 12*Dd[y,x]/CT1[x]
            P_agg[y,x] = 1 - P_U[y,x] - P_M[y,x] - P_Tr[y,x] - P_H[y,x] - P_Dd[y,x]
            #P_agg[y,x] = Agg[y,x]/CT1[x]
            P_sum[y,x] = P_U[y,x] + P_M[y,x] + P_Tr[y,x] + P_H[y,x] + P_Dd[y,x] + P_agg[y,x]
            
            #print(f"### WT ### T: {Temperatures[y]}, M: {M[y,x]}, Tr: {Tr[y,x]}, H: {H[y,x]}, Dd: {Dd[y,x]}")
            print(f"### WT ###: Temperature: {Temperatures[y]}\nK_M: {K_M[y,x]}\nK_Tr: {K_Tr[y,x]}\nK_H: {K_H[y,x]}\nK_Dd: {K_Dd[y,x]}\nPsum: {(M[y,x] + 3*Tr[y,x] + 6*H[y,x] + 12*Dd[y,x])/CT1[x]}\n")
    
    # Calculate dPdT for each state to use in calculating heat capacity
    for x in range(len(CT1)):
        
        dPUdT[:,x] = np.gradient(P_U[:,x])/np.gradient(Temperatures[:,0])
        dPMdT[:,x] = np.gradient(P_M[:,x])/np.gradient(Temperatures[:,0])
        dPTrdT[:,x] = np.gradient(P_Tr[:,x])/np.gradient(Temperatures[:,0])
        dPHdT[:,x] = np.gradient(P_H[:,x])/np.gradient(Temperatures[:,0])
        dPDddT[:,x] = np.gradient(P_Dd[:,x])/np.gradient(Temperatures[:,0])
        dPaggdT[:,x] = np.gradient(P_agg[:,x])/np.gradient(Temperatures[:,0])
        
        # Calculate heat capacity DSC profile           
        #Cp_MTHD[:,x] = Cp_U[:,x] + dPMdT[:,x]*dH_M[:,x] + dPTrdT[:,x]*dH_Tr[:,x] + dPHdT[:,x]*dH_H[:,x] + dPDddT[:,x]*dH_Dd[:,x] \# Temperature dependent dCps
        #+ P_M[:,x]*dCp_M[:,x] + P_Tr[:,x]*dCp_Tr[:,x] + P_H[:,x]*dCp_H[:,x] + P_Dd[:,x]*dCp_Dd[:,x]
        
        Cp_MTHD[:,x] = Cp_U[:,x] + dPMdT[:,x]*dH_M[:,x] + dPTrdT[:,x]*dH_Tr[:,x] + dPHdT[:,x]*dH_H[:,x] + dPDddT[:,x]*dH_Dd[:,x] \
        + P_M[:,x]*dCp_M + P_Tr[:,x]*dCp_Tr + P_H[:,x]*dCp_H + P_Dd[:,x]*dCp_Dd                                                   # Temperature independent dCps
    
    # Put populations into dictionary for plotting
    Pops_dict = {}
    Pops_dict['PU'] = P_U
    Pops_dict['PM'] = P_M
    Pops_dict['PTr'] = P_Tr
    Pops_dict['PH'] = P_H
    Pops_dict['PDd'] = P_Dd
    Pops_dict['Pagg'] = P_agg
    Pops_dict['Psum'] = P_sum
    
    return Cp_MTHD, Pops_dict

def MonTriHex(fit_params, Temperatures, CT0, Cp_U):
    
    # Set-up concentrations, parameters for calculating DSC Cp profiles
    CT1 = np.copy(CT0)
    CT_eq = np.copy(CT0)
    dH_M0 = fit_params['dH_M'].value
    dS_M0 = fit_params['dS_M'].value
    dH_Tr0 = fit_params['dH_Tr'].value
    dS_Tr0 = fit_params['dS_Tr'].value
    dH_H0 = fit_params['dH_H'].value
    dS_H0 = fit_params['dS_H'].value
    dH_agg0 = fit_params['dH_agg'].value
    kagg0 = fit_params['kagg0'].value
    Eagg = fit_params['Eagg'].value
    SR = fit_params['SR'].value
    T_inc = fit_params['T_inc'].value
    
    dCp_M = fit_params['dCp_M'].value
    dCp_Tr = fit_params['dCp_Tr'].value
    dCp_H = fit_params['dCp_H'].value

    U = np.zeros(len(Temperatures))
    M = np.zeros(len(Temperatures))
    Tr = np.zeros(len(Temperatures))
    H = np.zeros(len(Temperatures))
    Agg = np.zeros(len(Temperatures))

    P_U = np.zeros(len(Temperatures))
    P_M = np.zeros(len(Temperatures))
    P_Tr = np.zeros(len(Temperatures))
    P_H = np.zeros(len(Temperatures))
    P_agg = np.zeros(len(Temperatures))
    P_sum = np.zeros(len(Temperatures))
    
    dPMdT = np.zeros(len(Temperatures))
    dPTrdT = np.zeros(len(Temperatures))
    dPHdT = np.zeros(len(Temperatures))
    dPaggdT = np.zeros(len(Temperatures))
    
    dH_M = np.zeros(len(Temperatures))
    dH_Tr = np.zeros(len(Temperatures))
    dH_H = np.zeros(len(Temperatures))
    
    dS_M = np.zeros(len(Temperatures))
    dS_Tr = np.zeros(len(Temperatures))
    dS_H = np.zeros(len(Temperatures))
    
    dG_M = np.zeros(len(Temperatures))
    dG_Tr = np.zeros(len(Temperatures))
    dG_H = np.zeros(len(Temperatures))
    
    K_M = np.zeros(len(Temperatures))
    K_Tr = np.zeros(len(Temperatures))
    K_H = np.zeros(len(Temperatures))
    k_agg = np.zeros(len(Temperatures))
    
    Cp_MTH = np.zeros(len(Temperatures))
    
    # Loop through temperature and concentration data to calculate Cp
    for y in range(len(Temperatures)):
        
        # Calculate equilibrium constants            
        dH_M[y] = dH_M0 + (Temperatures[y] - Temperatures[0])*dCp_M             # Temperature-independent dCps
        dS_M[y] = dS_M0 + np.log(Temperatures[y]/Temperatures[0])*dCp_M
        dH_Tr[y] = dH_Tr0 + (Temperatures[y] - Temperatures[0])*dCp_Tr
        dS_Tr[y] = dS_Tr0 + np.log(Temperatures[y]/Temperatures[0])*dCp_Tr
        dH_H[y] = dH_H0 + (Temperatures[y] - Temperatures[0])*dCp_H
        dS_H[y] = dS_H0 + np.log(Temperatures[y]/Temperatures[0])*dCp_H
        
        dG_M[y] = dH_M[y] - Temperatures[y]*dS_M[y]
        dG_Tr[y] = dH_Tr[y] - Temperatures[y]*dS_Tr[y]
        dG_H[y] = dH_H[y] - Temperatures[y]*dS_H[y]
        
        K_M[y] = np.exp(-dG_M[y]/(R*Temperatures[y]))
        K_Tr[y] = np.exp(-dG_Tr[y]/(R*Temperatures[y]))
        K_H[y] = np.exp(-dG_H[y]/(R*Temperatures[y]))
        k_agg[y] = kagg0*np.exp((Eagg/R)*((1/Temperatures[0]) - (1/Temperatures[y])))
            
        # Get free monomer concentration from real, positive root of binding polynomial
        # Used to calculate concentrations of other oligomers
        rr = np.roots([6*K_H[y]*np.square(K_Tr[y])*np.power(K_M[y],6), 0, 0, 3*K_Tr[y]*np.power(K_M[y],3), 0, (K_M[y] + 1), -CT_eq])
            
        for root in rr:
            if np.isreal(root) and np.real(root) > 0:
                    
                U[y] = np.real(root)
                M[y] = K_M[y]*U[y]
                Tr[y] = K_Tr[y]*np.power(M[y],3)
                H[y] = K_H[y]*np.square(Tr[y])
                    
            # Aggregation kinetics - first point is just increase in aggregted state from the unfolded state
            # Assumes aggregation is first order kinetic process
            # The amount of aggregation is calculated as total unfolded minus how much unfolded remains after aggregation for a time t
            #if y == 0:
                #Agg[y] = U[y] - U[y]*np.exp(-k_agg[y]*T_inc/SR)
                    
            # Aggregation kinetics - point y is the sum of y-1 and the increase in aggregated state from the unfolded state
            # Assumes aggregation is first order kinetic process
            #else:
                #Agg[y] = Agg[y-1] + (U[y] - U[y]*np.exp(-k_agg[y]*T_inc/SR))
            
            # Aggregation kinetics - the total monomer concentration available
            # for equilibration between states gets knocked down every point
            # due to aggregation from the unfolded state
            #CT_eq = CT_eq - (U[y] - U[y]*np.exp(-k_agg[y]*T_inc/SR))
            
            # Populations of each state are calculated using the total concentration, not just the equilibration CT
            P_U[y] = U[y]/CT1
            P_M[y] = M[y]/CT1
            P_Tr[y] = 3*Tr[y]/CT1
            P_H[y] = 6*H[y]/CT1
            P_agg[y] = 1 - P_U[y] - P_M[y] - P_Tr[y] - P_H[y]
            P_sum[y] = P_U[y] + P_M[y] + P_Tr[y] + P_H[y] + P_agg[y]
            
            #print(f"### WT ### T: {Temperatures[y]}, M: {M[y,x]}, Tr: {Tr[y,x]}, H: {H[y,x]}, Dd: {Dd[y,x]}")
            #print(f"### WT ###: Temperature: {Temperatures[y]}\nK_M: {K_M[y]}\nK_Tr: {K_Tr[y]}\nK_H: {K_H[y]}\nPsum: {P_sum[y]}\n")
    
    # Calculate dPdT for each state to use in calculating heat capacity
    dPMdT = np.gradient(P_M)/np.gradient(Temperatures)
    dPTrdT = np.gradient(P_Tr)/np.gradient(Temperatures)
    dPHdT = np.gradient(P_H)/np.gradient(Temperatures)
    dPaggdT = np.gradient(P_agg)/np.gradient(Temperatures)
        
    Cp_MTH = Cp_U + dPMdT*dH_M + dPTrdT*dH_Tr + dPHdT*dH_H \
        + P_M*dCp_M + P_Tr*dCp_Tr + P_H*dCp_H                # Temperature independent dCps
    
    # Put populations into dictionary for plotting
    Pops_dict = {}
    Pops_dict['PU'] = P_U
    Pops_dict['PM'] = P_M
    Pops_dict['PTr'] = P_Tr
    Pops_dict['PH'] = P_H
    Pops_dict['Pagg'] = P_agg
    Pops_dict['Psum'] = P_sum
    
    return Cp_MTH, Pops_dict

def ThermoKinConstants(fit_params, Temperatures):
    
    dH_M0 = fit_params['dH_M0'].value
    dS_M0 = fit_params['dS_M0'].value
    dCp_M = fit_params['dCp_M'].value
    dH_Tr = fit_params['dH_Tr'].value
    dS_Tr = fit_params['dS_Tr'].value
    dH_H = fit_params['dH_H'].value
    dS_H = fit_params['dS_H'].value
    dH_Dd = fit_params['dH_Dd'].value
    dS_Dd = fit_params['dS_Dd'].value
    kagg0 = fit_params['kagg0'].value
    Eagg = fit_params['Eagg'].value
    
    dH_M = dH_M0 + (Temperatures - Temperatures[0])*dCp_M
    dS_M = dS_M0 + np.log(Temperatures/Temperatures[0])*dCp_M
    
    dG_M = dH_M - Temperatures*dS_M
    dG_Tr = dH_Tr - Temperatures*dS_Tr
    dG_H = dH_H - Temperatures*dS_H
    dG_Dd = dH_Dd - Temperatures*dS_Dd
    
    K_M = np.exp(-dG_M/(R*Temperatures))
    K_Tr = np.exp(-dG_Tr/(R*Temperatures))
    K_H = np.exp(-dG_H/(R*Temperatures))
    K_Dd = np.exp(-dG_Dd/(R*Temperatures))
    k_agg = kagg0*np.exp((Eagg/R)*((1/Temperatures[0]) - (1/Temperatures)))
    
    K_dict = {}
    K_dict['K_M'] = K_M
    K_dict['K_Tr'] = K_Tr
    K_dict['K_H'] = K_H
    K_dict['K_Dd'] = K_Dd
    K_dict['k_agg'] = k_agg
    
    return K_dict

def baselines(x, y, z, Temperatures):
    
    #Cp_F = m*(Temperatures - Temperatures[0]) + b
    Cp_U = z*np.square(Temperatures - Temperatures[0]) + y*(Temperatures - Temperatures[0]) + x
    
    return Cp_U

### Objective function for global fitting
#def objective(fit_params, P2_60uM, P2_665uM):
#    
#    # Set temperature ranges
#    a = np.argwhere((P2_60uM['Scan_5']['Temperature']>=20) & (P2_60uM['Scan_5']['Temperature']<=90))
#    P2_60uM_Temps = P2_60uM['Scan_5']['Temperature'][a] + 273.15
#    P2_60uM_Cp = P2_60uM['Scan_5']['Cp'][a]
#
#    a = np.argwhere((P2_665uM['Scan_11']['Temperature']>=20) & (P2_665uM['Scan_11']['Temperature']<=90))
#    P2_665uM_Temps = P2_665uM['Scan_11']['Temperature'][a]  + 273.15
#    P2_665uM_Cp = P2_665uM['Scan_11']['Cp'][a]
#    
#    #m1 = fit_params['m1'].value
#    #b1 = fit_params['b1'].value
#    x1 = fit_params['x1'].value
#    y1 = fit_params['y1'].value
#    z1 = fit_params['z1'].value
#    Cp_U = baselines(x1, y1, z1, P2_60uM_Temps)
#    Cp_60uM, Pdict_60uM = MTHD(fit_params, P2_60uM_Temps, [60e-6], Cp_U)
#    
#    fig, axs = plt.subplots(2,2,figsize=(11,6))
#    axs[0,0].plot(P2_60uM_Temps,P2_60uM_Cp,'ko')
#    axs[0,0].plot(P2_60uM_Temps,Cp_60uM,'r')
#    #axs[0,0,].plot(P2_60uM_Temps,Cp_F,'b')
#    axs[0,0].plot(P2_60uM_Temps,Cp_U,'g')
#    
#    axs[0,1].plot(P2_60uM_Temps,Pdict_60uM['PU'],'k',label='PU')
#    axs[0,1].plot(P2_60uM_Temps,Pdict_60uM['PM'],label='PM')
#    axs[0,1].plot(P2_60uM_Temps,Pdict_60uM['PTr'],label='PTr')
#    axs[0,1].plot(P2_60uM_Temps,Pdict_60uM['PH'],label='PH')
#    axs[0,1].plot(P2_60uM_Temps,Pdict_60uM['PDd'],label='PDd')
#    axs[0,1].plot(P2_60uM_Temps,Pdict_60uM['Pagg'],label='Pagg')
#    axs[0,1].plot(P2_60uM_Temps,Pdict_60uM['Psum'],label='Psum')
#    
#    #m2 = fit_params['m2'].value
#    #b2 = fit_params['b2'].value
#    x2 = fit_params['x2'].value
#    y2 = fit_params['y2'].value
#    z2 = fit_params['z2'].value
#    Cp_U = baselines(x2, y2, z2, P2_665uM_Temps)
#    Cp_665uM, Pdict_665uM = MTHD(fit_params, P2_665uM_Temps, [665e-6], Cp_U)
#    
#    axs[1,0].plot(P2_665uM_Temps,P2_665uM_Cp,'ko')
#    axs[1,0].plot(P2_665uM_Temps,Cp_665uM,'r')
#    #axs[1,0].plot(P2_665uM_Temps,Cp_F,'b')
#    axs[1,0].plot(P2_665uM_Temps,Cp_U,'g')
#    
#    axs[1,1].plot(P2_665uM_Temps,Pdict_665uM['PU'],'k',label='PU')
#    axs[1,1].plot(P2_665uM_Temps,Pdict_665uM['PM'],label='PM')
#    axs[1,1].plot(P2_665uM_Temps,Pdict_665uM['PTr'],label='PTr')
#    axs[1,1].plot(P2_665uM_Temps,Pdict_665uM['PH'],label='PH')
#    axs[1,1].plot(P2_665uM_Temps,Pdict_665uM['PDd'],label='PDd')
#    axs[1,1].plot(P2_665uM_Temps,Pdict_665uM['Pagg'],label='Pagg')
#    axs[1,1].plot(P2_665uM_Temps,Pdict_665uM['Psum'],label='Psum')
#    
#    plt.tight_layout()
#    plt.legend()
#    plt.ion()
#    plt.pause(0.01)
#    plt.ioff()
#    #plt.show()
#    plt.close()
#    
#    resid_60uM = P2_60uM_Cp - Cp_60uM
#    resid_665uM = P2_665uM_Cp - Cp_665uM
#    
#    resid = np.concatenate((resid_60uM,resid_665uM),axis=0)
#    #print('P2 expt: ',P2_60uM_Cp, '\n P2 sim: ',Cp_60uM)
#    #print(np.sum(resid))
#
#    return resid

### Objective function for global fitting
def objective(fit_params, Temperature, Cp, Concentration):

    x1 = fit_params['x1'].value
    y1 = fit_params['y1'].value
    z1 = fit_params['z1'].value
    Cp_U = baselines(x1, y1, z1, Temperature)
    Cp_sim, Pdict = MonTriHex(fit_params, Temperature, Concentration, Cp_U)
    
    fig, axs = plt.subplots(1,2,figsize=(11,4))
    axs[0].plot(Temperature,Cp,'ko')
    axs[0].plot(Temperature,Cp_sim,'r')
    axs[0].plot(Temperature,Cp_U,'g')
    
    axs[1].plot(Temperature,Pdict['PU'],'k',label='PU')
    axs[1].plot(Temperature,Pdict['PM'],label='PM')
    axs[1].plot(Temperature,Pdict['PTr'],label='PTr')
    axs[1].plot(Temperature,Pdict['PH'],label='PH')
    axs[1].plot(Temperature,Pdict['Pagg'],label='Pagg')
    axs[1].plot(Temperature,Pdict['Psum'],'k--',label='Psum')
    
    plt.tight_layout()
    plt.legend()
    plt.ion()
    plt.pause(0.00001)
    plt.ioff()
    plt.close()
    
    resid = Cp - Cp_sim

    return resid

##### Plot optimized fit result and save pdfs of plots for each Cp profile
#def plot_fit(fit_params, Temperatures, CT):
#    
#    # Open pdf to write figures to
#    pdf = matplotlib.backends.backend_pdf.PdfPages('DSCsimulations.pdf')
#    label_params = {'mathtext.default': 'regular' }          
#    plt.rcParams.update(label_params)
#    plt.rcParams['axes.linewidth'] = 2
#    colors = [plt.cm.jet(y) for y in range(150)]
#    color_idxs = np.linspace(0,100,len(CT))
#    
#    # Simulate Cp data based on input params, temperature, concentration
#    Cp_MTHD, Pops_dict = MTHD(fit_params, Temperatures, CT)
#    
#    # Calculate equilibrium and aggregation rate constants
#    K_dict = ThermoKinConstants(fit_params, Temperatures)
#    
#    # Plot equilibrium and aggregation rate constants
#    K_fig = plt.figure()
#    K_ax = K_fig.add_subplot(111)
#    K_ax.plot(1/Temperatures,np.log(K_dict['K_M']),LineWidth=2,label='$K_{M}$')
#    K_ax.plot(1/Temperatures,np.log(K_dict['K_Tr']),LineWidth=2,label='$K_{Tr}$')
#    K_ax.plot(1/Temperatures,np.log(K_dict['K_H']),LineWidth=2,label='$K_{H}$')
#    K_ax.plot(1/Temperatures,np.log(K_dict['K_Dd']),LineWidth=2,label='$K_{Dd}$')
#    K_ax.plot(1/Temperatures,np.log(K_dict['k_agg']),LineWidth=2,label='$k_{agg}$')
#    K_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#    K_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
#    K_ax.yaxis.major.formatter._useMathText = True
#    K_ax.set_title('Equilibrium and aggregation constants',fontsize=14)
#    K_ax.set_xlabel('1/Temperature $K^{-1}$',fontsize=14)
#    K_ax.set_ylabel('ln(K or k)',fontsize=14)
#    K_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
#    K_ax.legend()
#    #plt.show()
#    
#    # Plot data
#    for x,color_idx in zip(range(len(CT)),color_idxs):
#        
#        fig,axs = plt.subplots(1,2,figsize=(11,4))
#        # Plot Cp data
#        axs[0].plot(Temperatures-273.15,Cp_MTHD[:,x]/1000,LineWidth=2,color=colors[int(color_idx)])
#        #ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#        axs[0].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
#        axs[0].yaxis.major.formatter._useMathText = True
#        axs[0].set_title(f"$[M]_{0}$ = {np.round(CT[x]*1e6,1)} $\mu$M",fontsize=14)
#        axs[0].set_xlabel('Temperature \N{DEGREE SIGN}C',fontsize=14)
#        axs[0].set_ylabel('$C_{p}$ $kJ$ $mol^{-1}$ $K^{-1}$',fontsize=14)
#        axs[0].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
#        
#        # Plot populations
#        axs[1].plot(Temperatures-273.15,Pops_dict['PU'][:,x],'k',LineWidth=2,label='$P_{U}$')
#        axs[1].plot(Temperatures-273.15,Pops_dict['PM'][:,x],LineWidth=2,label='$P_{M}$')
#        axs[1].plot(Temperatures-273.15,Pops_dict['PTr'][:,x],LineWidth=2,label='$P_{Tr}$')
#        axs[1].plot(Temperatures-273.15,Pops_dict['PH'][:,x],LineWidth=2,label='$P_{H}$')
#        axs[1].plot(Temperatures-273.15,Pops_dict['PDd'][:,x],LineWidth=2,label='$P_{Dd}$')
#        axs[1].plot(Temperatures-273.15,Pops_dict['Pagg'][:,x],LineWidth=2,label='$P_{agg}$')
#        axs[1].plot(Temperatures-273.15,Pops_dict['Psum'][:,x],'k--',LineWidth=2)
#        #ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#        axs[1].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
#        axs[1].yaxis.major.formatter._useMathText = True
#        axs[1].set_title(f"$[M]_{0}$ = {np.round(CT[x]*1e6,1)} $\mu$M",fontsize=14)
#        axs[1].set_xlabel('Temperature \N{DEGREE SIGN}C',fontsize=14)
#        axs[1].set_ylabel('Population',fontsize=14)
#        axs[1].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
#        axs[1].legend()
#        
#        pdf.savefig(fig)
#        #plt.close()
#    
#    plt.show()
#    pdf.close()


#### Plot optimized fit result and save pdfs of plots for each Cp profile
def plot_fit(fit_params, Cp, Temperatures, CT):
    
    # Open pdf to write figures to
    pdf = matplotlib.backends.backend_pdf.PdfPages('DSCfit.pdf')
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    colors = [plt.cm.jet(y) for y in range(150)]
    color_idxs = np.linspace(0,100,len(CT))
    
    print(fit_params, Cp, Temperatures, CT)
    
    # Simulate Cp data based on input params, temperature, concentration
    Cp_U = baselines(fit_params['x1'].value, fit_params['y1'].value, fit_params['z1'].value, Temperatures)
    Cp_sim, Pops_dict = MonTriHex(fit_params, Temperatures, CT, Cp_U)
    
    # Plot data
    for x,color_idx in zip(range(len(CT)),color_idxs):
        
        fig,axs = plt.subplots(1,2,figsize=(11,4))
        # Plot Cp data
        axs[0].plot(Temperatures-273.15,Cp,'ko',LineWidth=2,color=colors[int(color_idx)])
        axs[0].plot(Temperatures-273.15,Cp_sim,'r',LineWidth=2)
        axs[0].plot(Temperatures-273.15,Cp_U,'g',LineWidth=2)
        axs[0].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        axs[0].yaxis.major.formatter._useMathText = True
        axs[0].set_title(f"$[M]_{0}$ = {np.round(CT[x]*1e6,1)} $\mu$M",fontsize=14)
        axs[0].set_xlabel('Temperature \N{DEGREE SIGN}C',fontsize=14)
        axs[0].set_ylabel('$C_{p}$ $kJ$ $mol^{-1}$ $K^{-1}$',fontsize=14)
        axs[0].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        
        # Plot populations
        axs[1].plot(Temperatures-273.15,Pops_dict['PU'],'k',LineWidth=2,label='$P_{U}$')
        axs[1].plot(Temperatures-273.15,Pops_dict['PM'],LineWidth=2,label='$P_{M}$')
        axs[1].plot(Temperatures-273.15,Pops_dict['PTr'],LineWidth=2,label='$P_{Tr}$')
        axs[1].plot(Temperatures-273.15,Pops_dict['PH'],LineWidth=2,label='$P_{H}$')
        axs[1].plot(Temperatures-273.15,Pops_dict['Pagg'],LineWidth=2,label='$P_{agg}$')
        axs[1].plot(Temperatures-273.15,Pops_dict['Psum'],'k--',LineWidth=2)
        axs[1].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        axs[1].yaxis.major.formatter._useMathText = True
        axs[1].set_title(f"$[M]_{0}$ = {np.round(CT[x]*1e6,1)} $\mu$M",fontsize=14)
        axs[1].set_xlabel('Temperature \N{DEGREE SIGN}C',fontsize=14)
        axs[1].set_ylabel('Population',fontsize=14)
        axs[1].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        axs[1].legend()
        
        pdf.savefig(fig)
        #plt.close()
    
    plt.show()
    pdf.close()
    
    
### Read in data and set up for fitting
def main(input_file):
    
    ### Set up parameters and groups
    # DegP_9 parameters
    #params = yaml.safe_load(open(sys.argv[1]))
    #P9_params = params['Samples']['P9_A9']   
    
    # DegP_2 60 uM params
    #pickle_in = open(sys.argv[1],"rb")
    #P2_60uM = pickle.load(pickle_in)
    
    #pickle_in = open(sys.argv[2],"rb")
    #P2_665uM = pickle.load(pickle_in)
    
    # Yuki hTRA2
    hTRA2 = pd.read_csv(sys.argv[1])
    hTRA2_Temperature = np.array(hTRA2['Temperature'][0::10]) + 273.15
    hTRA2_Cp = np.array(hTRA2['Cp kJ/molK'][0::10])
    hTRA2_Concentration = np.array([50e-6])
    
    fit_params = Parameters()
    # Endothermic hexamer and dodecamer params
    #fit_params.add('dH_M',value=-200)
    #fit_params.add('dS_M',value=-0.5)
    #fit_params.add('dH_Tr',value=-300)
    #fit_params.add('dS_Tr',value=-0.65)
    #fit_params.add('dH_H',value=-330)
    #fit_params.add('dS_H',value=-0.9)
    #fit_params.add('dH_Dd',value=-400)
    #fit_params.add('dS_Dd',value=-1.1)
    
    fit_params.add('dH_M',value=-200)
    fit_params.add('dS_M',value=-0.5)
    fit_params.add('dH_Tr',value=-400)
    fit_params.add('dS_Tr',value=-0.75)
    fit_params.add('dH_H',value=-600)
    fit_params.add('dS_H',value=-0.95)
    #fit_params.add('dH_Dd',value=40)
    #fit_params.add('dS_Dd',value=0.2)
    
    fit_params.add('dCp_M',value=-5)
    fit_params.add('dCp_Tr',value=-5)
    fit_params.add('dCp_H',value=-5)
    #fit_params.add('dCp_Dd',value=-2)
    fit_params.add('dH_agg',value=0,vary=False)
    fit_params.add('Eagg',value=0,vary=False)
    fit_params.add('kagg0',value=5e-12,vary=False)
    fit_params.add('SR',value=1/60,vary=False)
    #fit_params.add('T_inc',value=np.diff(P2_60uM['Scan_5']['Temperature'])[0],vary=False)
    fit_params.add('T_inc',value=np.diff(hTRA2_Temperature)[0],vary=False)
    #fit_params.add('m1',value=0.5)
    #fit_params.add('b1',value=-40)
    #fit_params.add('z1',value=0.01)
    #fit_params.add('y1',value=0.5)
    #fit_params.add('x1',value=-20)
    #fit_params.add('m2',value=0)
    #fit_params.add('b2',value=75)
    #fit_params.add('z2',value=0)
    #fit_params.add('y2',value=0)
    #fit_params.add('x2',value=110)
    #fit_params.add('m1',value=1)
    #fit_params.add('b1',value=-50)
    fit_params.add('z1',value=-0.004)
    fit_params.add('y1',value=0.3)
    fit_params.add('x1',value=80)
    #fit_params.add('m2',value=0)
    #fit_params.add('b2',value=80)
    #fit_params.add('z2',value=0)
    #fit_params.add('y2',value=0)
    #fit_params.add('x2',value=100)
    
    # DegP fit
    #result = minimize(objective, fit_params, method='nelder', args=(P2_60uM,P2_665uM))
    
    # hTRA2 fit
    result = minimize(objective, fit_params, method='nelder', args=(hTRA2_Temperature, hTRA2_Cp, hTRA2_Concentration))
    report_fit(result)
    
    #plot_fit(fit_params, Temperatures, CT_1)
    
    # hTRA2 plot fit
    plot_fit(result.params, hTRA2_Cp, hTRA2_Temperature, hTRA2_Concentration) # Plot fit result
    #plot_fit(fit_params, hTRA2_Cp, hTRA2_Temperature, hTRA2_Concentration) # Plot simulation result
    

### Run fit and generate result plots
R = 8.3145e-3 # Gas constant for calculating affinities
main(sys.argv[1])
