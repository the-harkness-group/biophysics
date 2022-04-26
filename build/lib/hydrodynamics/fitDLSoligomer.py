#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 21:07:34 2019

@author: robertharkness
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
from lmfit import minimize, Parameters, report_fit

## Model for monomer-trimer equilibrium
def MT(fit_params, group_Tr):
    
    # Set up arrays for calculating concentrations as a function of temperature and total monomer concentration
    Temperatures = np.array(group_Tr.Temperature + 273.15)
    Concentrations = np.array(group_Tr.Concentration*1e-6)
    
    M = np.zeros(len(Concentrations))
    Tr = np.zeros(len(Concentrations))
    Dz_MT = np.zeros(len(Concentrations))
    
    M_num = np.zeros(len(M))
    M_den = np.zeros(len(M))
    Tr_num = np.zeros(len(M))
    Tr_den = np.zeros(len(M))
    
    for x in range(len(Concentrations)):
        # Get free monomer concentration from real, positive root of binding polynomial
        # Used to calculate concentrations of other oligomers
        
        # Calculate equilibrium constants
        dG = fit_params['dH_Tr'].value - Temperatures[x]*fit_params['dS_Tr'].value
        #dG = fit_params['dG_Tr'].value - fit_params['dS_Tr'].value*(Temperatures[x] - 308.15)
        K = np.exp(-dG/(R*Temperatures[x]))

        rr = np.roots([3*K, 0, 1, -Concentrations[x]])
        
        for root in rr:
            if np.isreal(root) and np.real(root) > 0:
            
                M[x] = np.real(root)
                Tr[x] = (Concentrations[x] - M[x])/3
        
        # Calculate the numerator and denominator terms and finally the average diffusion coefficient
        D_M = StokesEinstein(Temperatures[x], fit_params['Rh_M'].value)
        D_Tr = StokesEinstein(Temperatures[x], fit_params['Rh_Tr'].value)
        
        M_num[x] = M[x]*D_M*(1 + fit_params['k_c'].value*M[x])
        Tr_num[x] = 9*Tr[x]*D_Tr*(1 + fit_params['k_c'].value*Tr[x])
            
        M_den[x] = M[x]
        Tr_den[x] = 9*Tr[x]
            
        Dz_MT[x] = (M_num[x] + Tr_num[x])/(M_den[x] + Tr_den[x])
        
        #print(f"################### TRIMER ####################### \nT: {Temperatures[x]}, M: {M[x]}, Tr: {Tr[x]}, D_M: {D_M}, D_Tr: {D_Tr}, Dz_MT: {Dz_MT[x]}\nKTr: {K}, Populations: PM: {M[x]/Concentrations[x]}, PTr: {3*Tr[x]/Concentrations[x]}, Ptot: {(M[x] + 3*Tr[x])/Concentrations[x]}\n##############################################\n")

    return Dz_MT

def Trimer(fit_params, group_Tr):
    
    # Get temperatures and concentrations out of pandas groups
    Temperatures = np.array(group_Tr.Temperature) + 273.15
    Concentrations = np.array(group_Tr.Concentration*1e-6)

    # Initialize z-average diffusion coefficient array
    Dz_Tr = np.zeros(len(Concentrations))
    
    for x in range(len(Concentrations)):
        
        # Calculate ideal trimer diffusion coefficient
        D_Tr = StokesEinstein(Temperatures[x], fit_params['Rh_Tr'].value)
        
        # Calculate z-average diffusion coefficient
        Dz_Tr[x] = D_Tr*(1 + fit_params['k_c'].value*Concentrations[x])
    
    return Dz_Tr

## Model for monomer-trimer-hexamer-dodecamer equilibrium
def MTHD(fit_params, group_WT):
    
    # Set up arrays for calculating concentrations as a function of temperature and total monomer concentration
    Temperatures = np.array(group_WT.Temperature + 273.15)
    Concentrations = np.array(group_WT.Concentration*1e-6)
    
    M = np.zeros(len(Concentrations))
    Tr = np.zeros(len(Concentrations))
    H = np.zeros(len(Concentrations))
    Dd = np.zeros(len(Concentrations))
    
    M_num = np.zeros(len(Concentrations))
    Tr_num = np.zeros(len(Concentrations))
    H_num = np.zeros(len(Concentrations))
    Dd_num = np.zeros(len(Concentrations))
    
    M_den = np.zeros(len(Concentrations))
    Tr_den = np.zeros(len(Concentrations))
    H_den = np.zeros(len(Concentrations))
    Dd_den = np.zeros(len(Concentrations))
    
    Dz_MTHD = np.zeros(len(Concentrations))

    for x in range(len(Concentrations)):
            
            # Calculate equilibrium constants
            dG_Tr = fit_params['dH_Tr'].value - Temperatures[x]*fit_params['dS_Tr'].value
            dG_H = fit_params['dH_H'].value - Temperatures[x]*fit_params['dS_H'].value
            dG_Dd = fit_params['dH_Dd'].value - Temperatures[x]*fit_params['dS_Dd'].value
            
#            dG_Tr = fit_params['dG_Tr'].value - fit_params['dS_Tr'].value*(Temperatures[x] - 308.15)
#            dG_H = fit_params['dG_H'].value - fit_params['dS_H'].value*(Temperatures[x] - 308.15)
#            dG_Dd = fit_params['dG_Dd'].value - fit_params['dS_Dd'].value*(Temperatures[x] - 308.15)

            K_Tr = np.exp(-dG_Tr/(R*Temperatures[x]))
            K_H = np.exp(-dG_H/(R*Temperatures[x]))
            K_Dd = np.exp(-dG_Dd/(R*Temperatures[x]))
            
            # Get free monomer concentration from real, positive root of binding polynomial
            # Used to calculate concentrations of other oligomers
            rr = np.roots([12*K_Dd*np.square(K_H)*np.power(K_Tr,4), 0, 0, 0, 0, 0, 6*K_H*np.square(K_Tr), 0, 0, 3*K_Tr, 0, 1, -Concentrations[x]])
            
            for root in rr:
                if np.isreal(root) and np.real(root) > 0:
            
                    M[x] = np.real(root)
                    Tr[x] = K_Tr*np.power(M[x],3)
                    H[x] = K_H*np.square(Tr[x])
                    Dd[x] = K_Dd*np.square(H[x])
            
            # Calculate the numerator and denominator terms and finally the average diffusion coefficient
            D_M = StokesEinstein(Temperatures[x], fit_params['Rh_M'].value)
            D_Tr = StokesEinstein(Temperatures[x], fit_params['Rh_Tr'].value)
            D_H = StokesEinstein(Temperatures[x], fit_params['Rh_H'].value)
            D_Dd = StokesEinstein(Temperatures[x], fit_params['Rh_Dd'].value)
            
            M_num[x] = M[x]*D_M*(1 + fit_params['k_c'].value*M[x])
            Tr_num[x] = 9*Tr[x]*D_Tr*(1 + fit_params['k_c'].value*Tr[x])
            H_num[x] = 36*H[x]*D_H*(1 + fit_params['k_c'].value*H[x])
            Dd_num[x] = 144*Dd[x]*D_Dd*(1 + fit_params['k_c'].value*Dd[x])
            
            M_den[x] = M[x]
            Tr_den[x] = 9*Tr[x]
            H_den[x] = 36*H[x]
            Dd_den[x] = 144*Dd[x]
            
            Dz_MTHD[x] = (M_num[x] + Tr_num[x] + H_num[x] + Dd_num[x])/(M_den[x] + Tr_den[x] + H_den[x] + Dd_den[x])

            #print(f"### WT ### T: {Temperatures[x]}, M: {M[x]}, Tr: {Tr[x]}, H: {H[x]}, Dd: {Dd[x]}, Dz_MTHD: {Dz_MTHD[x]}")
            #print(f"### WT ###: K_Tr: {K_Tr}, K_H: {K_H}, K_Dd: {K_Dd}, Psum: {(M[x] + 3*Tr[x] + 6*H[x] + 12*Dd[x])/Concentrations[x]}")

    return Dz_MTHD

def THD(fit_params, group_WT):
    
    # Set up arrays for calculating concentrations as a function of temperature and total monomer concentration
    Temperatures = np.array(group_WT.Temperature + 273.15)
    Concentrations = np.array(group_WT.Concentration*1e-6)
    
    Tr = np.zeros(len(Concentrations))
    H = np.zeros(len(Concentrations))
    Dd = np.zeros(len(Concentrations))
    
    Tr_num = np.zeros(len(Concentrations))
    H_num = np.zeros(len(Concentrations))
    Dd_num = np.zeros(len(Concentrations))
    
    Tr_den = np.zeros(len(Concentrations))
    H_den = np.zeros(len(Concentrations))
    Dd_den = np.zeros(len(Concentrations))
    
    Dz_THD = np.zeros(len(Concentrations))
    
    P_dict = {'PTr':[],'PH':[],'PDd':[]}
    #print(P_dict)
    
    dH_H = fit_params['dH_H'].value
    dS_H = fit_params['dS_H'].value
    dH_Dd = fit_params['dH_Dd'].value
    dS_Dd = fit_params['dS_Dd'].value

    for x in range(len(Concentrations)):
            
            # Calculate equilibrium constants
            dG_H = fit_params['dH_H'].value - Temperatures[x]*fit_params['dS_H'].value
            dG_Dd = fit_params['dH_Dd'].value - Temperatures[x]*fit_params['dS_Dd'].value
            
#            dG_Tr = fit_params['dG_Tr'].value - fit_params['dS_Tr'].value*(Temperatures[x] - 308.15)
#            dG_H = fit_params['dG_H'].value - fit_params['dS_H'].value*(Temperatures[x] - 308.15)
#            dG_Dd = fit_params['dG_Dd'].value - fit_params['dS_Dd'].value*(Temperatures[x] - 308.15)

            K_H = np.exp(-dG_H/(R*Temperatures[x]))
            K_Dd = np.exp(-dG_Dd/(R*Temperatures[x]))
            k_c = fit_params['m'].value*(Temperatures[x] - fit_params['T0'].value) + fit_params['k_c0'].value
            
            print(f"############### TRIMER-HEXAMER-DODECAMER MODEL ###################\nTemperature: {Temperatures[x]}\ndH_H: {dH_H}\ndS_H: {dS_H}\nK_H: {K_H}\ndH_Dd: {dH_Dd}\ndS_Dd: {dS_Dd}\nK_Dd: {K_Dd}\n")
            
            # Get free monomer concentration from real, positive root of binding polynomial
            # Used to calculate concentrations of other oligomers
            rr = np.roots([12*K_Dd*np.square(K_H), 0, 6*K_H, 3, -Concentrations[x]])
            
            for root in rr:
                if np.isreal(root) and np.real(root) > 0:
            
                    Tr[x] = np.real(root)
                    #Tr[x] = K_Tr*np.power(M[x],3)
                    H[x] = K_H*np.square(Tr[x])
                    Dd[x] = K_Dd*np.square(H[x])
            
            # Calculate the numerator and denominator terms and finally the average diffusion coefficient
            D_Tr = StokesEinstein(Temperatures[x], fit_params['Rh_Tr'].value)
            D_H = StokesEinstein(Temperatures[x], fit_params['Rh_H'].value)
            D_Dd = StokesEinstein(Temperatures[x], fit_params['Rh_Dd'].value)
            
            #Tr_num[x] = Tr[x]*(D_Tr*(1 + fit_params['k_c'].value*Tr[x]))
            #H_num[x] = 4*H[x]*(D_H*(1 + fit_params['k_c'].value*H[x]))
            #Dd_num[x] = 16*Dd[x]*(D_Dd*(1 + fit_params['k_c'].value*Dd[x]))
            
            Tr_num[x] = Tr[x]*(D_Tr*(1 + k_c*Tr[x]))
            H_num[x] = 4*H[x]*(D_H*(1 + k_c*H[x]))
            Dd_num[x] = 16*Dd[x]*(D_Dd*(1 + k_c*Dd[x]))
            
            Tr_den[x] = Tr[x]
            H_den[x] = 4*H[x]
            Dd_den[x] = 16*Dd[x]
            
            Dz_THD[x] = (Tr_num[x] + H_num[x] + Dd_num[x])/(Tr_den[x] + H_den[x] + Dd_den[x])
            
            #print(f"Temperature:{Temperatures[x]}, Concentration:{Concentrations[x]}")
            
            # Make dictionary of the populations
            P_dict['PTr'].append(3*Tr[x]/Concentrations[x])
            P_dict['PH'].append(6*H[x]/Concentrations[x])
            P_dict['PDd'].append(12*Dd[x]/Concentrations[x])

            #print(f"### WT ### T: {Temperatures[x]}, M: {M[x]}, Tr: {Tr[x]}, H: {H[x]}, Dd: {Dd[x]}, Dz_MTHD: {Dz_MTHD[x]}")
            #print(f"### WT ###\nK_H: {K_H}\nK_Dd: {K_Dd}\nPsum: {(3*Tr[x] + 6*H[x] + 12*Dd[x])/Concentrations[x]}\nTemperature: {Temperatures[x]}\n")
            #print(P_dict,'Concentrations:',Concentrations[x])
            
    return Dz_THD, P_dict

def THD24(fit_params, group_WT):
    
    # Set up arrays for calculating concentrations as a function of temperature and total monomer concentration
    Temperatures = np.array(group_WT.Temperature + 273.15)
    Concentrations = np.array(group_WT.Concentration*1e-6)
    
    Tr = np.zeros(len(Concentrations))
    H = np.zeros(len(Concentrations))
    Dd = np.zeros(len(Concentrations))
    Didd = np.zeros(len(Concentrations))
    
    Tr_num = np.zeros(len(Concentrations))
    H_num = np.zeros(len(Concentrations))
    Dd_num = np.zeros(len(Concentrations))
    Didd_num = np.zeros(len(Concentrations))
    
    Tr_den = np.zeros(len(Concentrations))
    H_den = np.zeros(len(Concentrations))
    Dd_den = np.zeros(len(Concentrations))
    Didd_den = np.zeros(len(Concentrations))
    
    Dz_THD24 = np.zeros(len(Concentrations))
    
    P_dict = {'PTr':[],'PH':[],'PDd':[],'P24':[]}
    #print(P_dict)
    
    dH_H = fit_params['dH_H'].value
    K0_H= fit_params['K0_H'].value
    dH_Dd = fit_params['dH_Dd'].value
    K0_Dd = fit_params['K0_Dd'].value
    dH_24 = fit_params['dH_24'].value
    K0_24 = fit_params['K0_24'].value

    for x in range(len(Concentrations)):
            
            # Calculate equilibrium constants
            K_H = K0_H*np.exp((-dH_H/R)*((1/Temperatures[x]) - (1./fit_params['T0'].value)))
            K_Dd = K0_Dd*np.exp((-dH_Dd/R)*((1/Temperatures[x]) - (1./fit_params['T0'].value)))
            K_24 = K0_24*np.exp((-dH_24/R)*((1/Temperatures[x]) - (1./fit_params['T0'].value)))

            k_c = fit_params['m'].value*(Temperatures[x] - fit_params['T0'].value) + fit_params['k_c0'].value
            
            #print(f"############### TRIMER-HEXAMER-DODECAMER-24MER MODEL ###################\nTemperature: {Temperatures[x]}\ndH_H: {dH_H}\nK_H: {K_H}\ndH_Dd: {dH_Dd}\ndK_Dd: {K_Dd}\ndH_24: {dH_24}\nK_24: {K_24}\n")
            
            # Get free monomer concentration from real, positive root of binding polynomial
            # Used to calculate concentrations of other oligomers
            rr = np.roots([24*K_24*np.square(K_Dd)*np.power(K_H,4), 0, 0, 0, 12*K_Dd*np.square(K_H), 0, 6*K_H, 3, -Concentrations[x]])
            
            for root in rr:
                if np.isreal(root) and np.real(root) > 0:
            
                    Tr[x] = np.real(root)
                    #Tr[x] = K_Tr*np.power(M[x],3)
                    H[x] = K_H*np.square(Tr[x])
                    Dd[x] = K_Dd*np.square(H[x])
                    Didd[x] = K_24*np.square(Dd[x])
            
            # Calculate the numerator and denominator terms and finally the average diffusion coefficient
            D_Tr = StokesEinstein(Temperatures[x], fit_params['Rh_Tr'].value)
            D_H = StokesEinstein(Temperatures[x], fit_params['Rh_H'].value)
            D_Dd = StokesEinstein(Temperatures[x], fit_params['Rh_Dd'].value)
            D_24 = StokesEinstein(Temperatures[x], fit_params['Rh_24'].value)
            
            Tr_num[x] = Tr[x]*(D_Tr*(1 + k_c*Tr[x]))
            H_num[x] = 4*H[x]*(D_H*(1 + k_c*H[x]))
            Dd_num[x] = 16*Dd[x]*(D_Dd*(1 + k_c*Dd[x]))
            Didd_num[x] = 64*Didd[x]*(D_24*(1 + k_c*Didd[x]))
            
            Tr_den[x] = Tr[x]
            H_den[x] = 4*H[x]
            Dd_den[x] = 16*Dd[x]
            Didd_den[x] = 64*Didd[x]
            
            Dz_THD24[x] = (Tr_num[x] + H_num[x] + Dd_num[x] + Didd_num[x])/(Tr_den[x] + H_den[x] + Dd_den[x] + Didd_den[x])
            
            #print(f"Temperature:{Temperatures[x]}, Concentration:{Concentrations[x]}")
            
            # Make dictionary of the populations
            P_dict['PTr'].append(3*Tr[x]/Concentrations[x])
            P_dict['PH'].append(6*H[x]/Concentrations[x])
            P_dict['PDd'].append(12*Dd[x]/Concentrations[x])
            P_dict['P24'].append(24*Didd[x]/Concentrations[x])

            #print(f"### WT ### T: {Temperatures[x]}, M: {M[x]}, Tr: {Tr[x]}, H: {H[x]}, Dd: {Dd[x]}, Dz_MTHD: {Dz_MTHD[x]}")
            #print(f"### WT ###\nK_H: {K_H}\nK_Dd: {K_Dd}\nPsum: {(3*Tr[x] + 6*H[x] + 12*Dd[x])/Concentrations[x]}\nTemperature: {Temperatures[x]}\n")
            #print(P_dict,'Concentrations:',Concentrations[x])
            
    return Dz_THD24, P_dict

### Model for diffusion coefficients of a trimer-hexamer system
def TrimerHexamer(fit_params, group_WT):
    
    # Set up arrays for calculating concentrations as a function of temperature and total monomer concentration
    Temperatures = np.array(group_WT.Temperature) + 273.15
    Concentrations = np.array(group_WT.Concentration)*1e-6
    
    Tr = np.zeros(len(Concentrations))
    H = np.zeros(len(Concentrations))
    
    # Numerator and denominator of diffusion coefficient equation
    Tr_num = np.zeros(len(Concentrations))
    H_num = np.zeros(len(Concentrations))
    Tr_den = np.zeros(len(Concentrations))
    H_den = np.zeros(len(Concentrations))
    Dz_TH = np.zeros(len(Concentrations))
    
    # Population dictionary for trimer and hexamer
    P_dict = {'PTr':[],'PH':[]}

    for x in range(len(Concentrations)):
            
    # Calculate equilibrium constants
        dG_H = fit_params['dH_H'].value - Temperatures[x]*fit_params['dS_H'].value
        K_H = np.exp(-dG_H/(R*Temperatures[x]))
            
    # Get free Trimer concentration from quadratic equation since this is 2Tr <-> Tr2 ie H
        Tr[x] = (-3 + np.sqrt(9 + 24*K_H*Concentrations[x]))/(12*K_H)    
        H[x] = (Concentrations[x] - 3*Tr[x])/6
        
        print(f"CT: {Concentrations[x]}\nTemperature: {Temperatures[x]}\nTrimer concentration: {Tr[x]}\nHexamer concentration: {H[x]}")

    # Calculate the numerator and denominator terms and finally the average diffusion coefficient
        D_Tr = StokesEinstein(Temperatures[x], fit_params['Rh_Tr'].value)
        D_H = StokesEinstein(Temperatures[x], fit_params['Rh_H'].value)
            
        Tr_num[x] = Tr[x]*D_Tr*(1 + fit_params['k_c'].value*Tr[x])
        H_num[x] = 4*H[x]*D_H*(1 + fit_params['k_c'].value*H[x])
            
        Tr_den[x] = Tr[x]
        H_den[x] = 4*H[x]
            
        Dz_TH[x] = (Tr_num[x] + H_num[x])/(Tr_den[x] + H_den[x])
            
    # Make dictionary of the populations
        P_dict['PTr'].append(3*Tr[x]/Concentrations[x])
        P_dict['PH'].append(6*H[x]/Concentrations[x])
            
    return Dz_TH, P_dict

### Calculate diffusion coefficients
def StokesEinstein(T, Rh):
    
    kB = 1.38065e-23
    eta = (2.414e-5)*10**(247.8/(T-140))
    Dt = (kB*T)/(6*np.pi*eta*Rh)
    
    return Dt
    
### Minimization function for global fit
def objective(fit_params, x, data_WT):
    
    #resid_MT = []
    #resid_Tr = []
    resid_THD24 = []
#    resid_TH = []
#    for ind, group in data_Tr:
#        Dz_Tr = Trimer(fit_params, group)*1e4
#        for x,y in enumerate(Dz_Tr):
#            resid_Tr.append((np.array(group.D.values[x]*1e4) - Dz_Tr[x]))
    
    for ind, group in data_WT:
        #Dz_MTHD = MTHD(fit_params, group)*1e4
        Dz_THD24,P_dict = THD24(fit_params, group)
        for x,y in enumerate(Dz_THD24):
            resid_THD24.append(np.array(group.D.values[x]*1e4) - Dz_THD24[x]*1e4)
            
    #fig, axs = plt.subplots(1,2,figsize=(11,4))
    #axs[0].plot(Temperature,Cp,'ko')
    #axs[0].plot(Temperature,Cp_sim,'r')
    #axs[0].plot(Temperature,Cp_U,'g')
    
    #axs[1].plot(Temperature,Pdict['PTr'],label='PTr')
    #axs[1].plot(Temperature,Pdict['PH'],label='PH')
    #axs[1].plot(Temperature,Pdict['PDd'],label='PDd')
    
    #plt.tight_layout()
    #plt.legend()
    #plt.ion()
    #plt.pause(0.00001)
    #plt.ioff()
    #plt.close()
    
    resid = resid_THD24
    
    return resid

#### Plot optimized fit result and save pdfs of plots for each temperature
#def plot_fit(data_Tr, data_WT, opt_params, CT_MTsim, CT_MTHDsim):
#    
#    pdf = matplotlib.backends.backend_pdf.PdfPages('DLSglobalfits.pdf')
#    label_params = {'mathtext.default': 'regular' }          
#    plt.rcParams.update(label_params)
#    plt.rcParams['axes.linewidth'] = 2
#    colors = [plt.cm.plasma(y) for y in range(150)]
#    
#    for ind, group in data_Tr:
#        
#        Tr_sim = pd.DataFrame()
#        Tr_sim['Temperature'] = np.zeros(len(CT_MTsim))
#        Tr_sim['Temperature'][:] = np.array(group['Temperature'])[0]
#        Tr_sim['Concentration'] = CT_MTsim
#
#        #Dz_Tr = Trimer(opt_params.params, Tr_sim)
#        Dz_Tr = Trimer(opt_params, Tr_sim)
#        Temperatures = np.array(group.Temperature)
#        
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.plot(group.Concentration,group.D*1e4,'bo')
#        ax.plot(Tr_sim['Concentration'],Dz_Tr*1e4,'r',LineWidth=2)
#        ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#        ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
#        ax.yaxis.major.formatter._useMathText = True
#        ax.set_title(f"DegP_5, Temperature: {np.round(Temperatures[0])} \N{DEGREE SIGN}C",fontsize=14)
#        ax.set_xlabel('$[M]_T$ $\mu$M',fontsize=14)
#        ax.set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$',fontsize=14)
#        ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
#    
#        pdf.savefig(fig)
#        plt.close()
#        
#    for ind, group in data_WT:
#        
#        WT_sim = pd.DataFrame()
#        WT_sim['Temperature'] = np.zeros(len(CT_MTHDsim))
#        WT_sim['Temperature'] = np.array(group['Temperature'])[0]
#        WT_sim['Concentration'] = CT_MTHDsim
#        
#        #Dz_TH, P_dict = TrimerHexamer(opt_params.params, WT_sim)
#        Dz_TH, P_dict = TrimerHexamer(opt_params, WT_sim)
#        Temperatures = np.array(group.Temperature)
#        
#        fig, axs = plt.subplots(1,2,figsize=(11,4))
#        axs[0].plot(group.Concentration,group.D*1e4,'ko')
#        axs[0].plot(WT_sim['Concentration'],Dz_TH*1e4,'r',LineWidth=2)
#        axs[0].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#        axs[0].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
#        axs[0].yaxis.major.formatter._useMathText = True
#        axs[0].set_title(f"DegP_2, Temperature: {np.round(Temperatures[0])} \N{DEGREE SIGN}C",fontsize=14)
#        axs[0].set_xlabel('$[M]_T$ $\mu$M',fontsize=14)
#        axs[0].set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$',fontsize=14)
#        axs[0].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
#        
#        axs[1].plot(WT_sim['Concentration'],P_dict['PTr'],LineWidth=2,label='$P_{Tr}$')
#        axs[1].plot(WT_sim['Concentration'],P_dict['PH'],LineWidth=2,label='$P_{H}$')
#        #axs[1].plot(group.Concentration,P_dict['PDd'],LineWidth=2,label='$P_{Dd}$')
#        axs[1].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#        axs[1].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
#        axs[1].yaxis.major.formatter._useMathText = True
#        axs[1].set_title(f"DegP_2, Temperature: {np.round(Temperatures[0])} \N{DEGREE SIGN}C",fontsize=14)
#        axs[1].set_xlabel('$[M]_T$ $\mu$M',fontsize=14)
#        axs[1].set_ylabel('Population',fontsize=14)
#        axs[1].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
#        axs[1].legend(loc='upper left',fontsize=12,frameon=False)
#        
#        pdf.savefig(fig)
#        plt.close()
#        
#    pdf.close()
    
### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(data_WT, opt_params, CT_THDsim):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('DLSglobalfits.pdf')
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2
    #colors = [plt.cm.plasma(y) for y in range(150)]
    
    for ind, group in data_WT:
        
        WT_sim = pd.DataFrame()
        WT_sim['Temperature'] = np.zeros(len(CT_THDsim))
        WT_sim['Temperature'] = np.array(group['Temperature'])[0]
        WT_sim['Concentration'] = CT_THDsim
        
        #Dz_TH, P_dict = TrimerHexamer(opt_params.params, WT_sim)
        #Dz_TH, P_dict = TrimerHexamer(opt_params, WT_sim)
        Dz_THD24, P_dict = THD24(opt_params, WT_sim)
        Temperatures = np.array(group.Temperature)
        
        fig, axs = plt.subplots(1,2,figsize=(11,6))
        axs[0].plot(group.Concentration,group.D*1e4,'ko')
        axs[0].plot(WT_sim['Concentration'],Dz_THD24*1e4,'r',LineWidth=2)
        axs[0].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        axs[0].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        axs[0].yaxis.major.formatter._useMathText = True
        axs[0].set_title(f"DegP_2, Temperature: {np.round(Temperatures[0])} \N{DEGREE SIGN}C",fontsize=14)
        axs[0].set_xlabel('$[M]_T$ $\mu$M',fontsize=14)
        axs[0].set_ylabel('$D_{z}$ $cm^{2}$ $s^{-1}$',fontsize=14)
        axs[0].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        
        axs[1].plot(WT_sim['Concentration'],P_dict['PTr'],LineWidth=2,label='$P_{Tr}$')
        axs[1].plot(WT_sim['Concentration'],P_dict['PH'],LineWidth=2,label='$P_{H}$')
        axs[1].plot(WT_sim['Concentration'],P_dict['PDd'],LineWidth=2,label='$P_{Dd}$')
        axs[1].plot(WT_sim['Concentration'],P_dict['P24'],LineWidth=2,label='$P_{24}$')
        axs[1].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        axs[1].tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        axs[1].yaxis.major.formatter._useMathText = True
        axs[1].set_title(f"DegP_2, Temperature: {np.round(Temperatures[0])} \N{DEGREE SIGN}C",fontsize=14)
        axs[1].set_xlabel('$[M]_T$ $\mu$M',fontsize=14)
        axs[1].set_ylabel('Population',fontsize=14)
        axs[1].grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        axs[1].legend(loc='upper left',fontsize=12,frameon=False)
        
        PD_fig = plt.figure()
        PD_ax = PD_fig.add_subplot(111)
        PD_ax.plot(group.Concentration,group.PD*100,'ko')
        PD_ax.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
        PD_ax.yaxis.major.formatter._useMathText = True
        PD_ax.set_title(f"DegP_2, Temperature: {np.round(Temperatures[0])} \N{DEGREE SIGN}C",fontsize=14)
        PD_ax.set_xlabel('$[M]_T$ $\mu$M',fontsize=14)
        PD_ax.set_ylabel('Polydispersity %',fontsize=14)
        PD_ax.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
        #PD_ax.legend(loc='upper left',fontsize=12,frameon=False)
        
        pdf.savefig(fig)
        pdf.savefig(PD_fig)
        plt.close()
        
    pdf.close()

### Read in data and set up for fitting
def main():
    data = pd.read_csv(sys.argv[1])

    ### Set up parameters and groups    
    #data_Tr = data[data['Sample'] == 'P5'] # DegP trimer mutant
    #data_Tr = data_Tr[data_Tr['Temperature'] < 48]
    #data_Tr = data_Tr[data_Tr['Concentration'] > 10]
    data_WT = data[data['Sample'] == 'P2'] # DegP protease-dead, WT
    #data_WT = data_WT[data_WT['Temperature'] < 48]
    data_WT = data_WT[data_WT['Concentration'] > 5]
    #data_WT = data_WT[data_WT['PD'] <=0.3]
    
    #data_Tr.to_csv('DegP5_DLSparams_20190507.csv')
    #data_WT.to_csv('DegP2_DLSparams_20190507.csv')
    
    #groups_Tr = data_Tr.groupby('Temperature')
    groups_WT = data_WT.groupby('Temperature')
    
    #CT_MTsim = np.linspace(np.max(data_Tr.Concentration),0.1,100)
    CT_MTHDsim = np.linspace(np.max(data_WT.Concentration),0.1,100)
    
    fit_params = Parameters()
    fit_params.add('dH_H',value=-50000)
    fit_params.add('K0_H',value=1e7,min=0,max=1e15)
    fit_params.add('dH_Dd',value=100000)
    fit_params.add('K0_Dd',value=1e3,min=0,max=1e10)
    fit_params.add('dH_24',value=400000)
    fit_params.add('K0_24',value=1e-2,min=0,max=1e10)
    
    #fit_params.add('dH_Tr',value=-400000,max=0)
    #fit_params.add('dS_Tr',value=-1100,max=0)
    #fit_params.add('dH_H',value=-100000,max=0)
    #fit_params.add('dS_H',value=-200,max=0)
    #fit_params.add('dH_Dd',value=100000,min=0)
    #fit_params.add('dS_Dd',value=600,min=0)

    #fit_params.add('Rh_M',value=0,vary=False,min=0)
    fit_params.add('Rh_Tr',value=4.7e-9,vary=False)
    fit_params.add('Rh_H',value=5.8e-9,vary=False)
    fit_params.add('Rh_Dd',value=8.8e-9,vary=False)
    fit_params.add('Rh_24',value=15e-9,vary=False)
    fit_params.add('k_c0',value=-3000,max=0)
    fit_params.add('m',value=-3)
    fit_params.add('T0',value=278.15,vary=False)
    
    # Fit data
    #result = minimize(objective, fit_params, method='nelder', args=(1, data_WT))
    #result = minimize(objective, fit_params, method='nelder', args=(1, groups_WT))
    #report_fit(result)
    
    # Simulate data
    #plot_fit(groups_Tr, groups_WT, result, CT_MTsim, CT_MTHDsim)
    #plot_fit(groups_Tr, groups_WT, fit_params, CT_MTsim, CT_MTHDsim)
    plot_fit(groups_WT, fit_params, CT_MTHDsim)

### Run fit and generate result plots
R = 1.9872e-3 # Gas constant for calculating affinities
R = R*4.184*1000 # Thermodynamic parameters have to be in J/mol and J/mol*K to match this gas constant
main()



