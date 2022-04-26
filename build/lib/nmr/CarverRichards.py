#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:19:13 2020

@author: robertharkness
"""

import numpy as np
import matplotlib.pyplot as plt

def main():

    vcpmg = np.linspace(50,1000,100) # Hz
    
    K = 100
    MT = 1e-3
    kab = 53000
    kba = kab/K
    R2a = 5
    R2b = 5
    dw = 80*1*2.*np.pi # rad/s
    
    pa, pb, M, M2 = Populations(K, MT)    
    R2eff = CarverRichards(pa, pb, M, kab, kba, R2a, R2b, dw, vcpmg)    
    PlotR2eff(vcpmg, R2eff)
    print(f"PA: {pa}, PB: {pb}, kex: {2*kab*M + kba}")
    
def Populations(K, MT):
    
    M = (-1 + np.sqrt(1 + 8.*K*MT))/(4.*K)
    M2 = (MT - M)/2.
    
    pa = M/MT
    pb = 2.*M2/MT
    
    return pa, pb, M, M2

def CarverRichards(pa, pb, M, kab, kba, R2a, R2b, dw, vcpmg):
    
    tcp = (1/(2*vcpmg))
    
    kex = 2*kab*M + kba
    pb = kba/kex
    pa = 2*kab*M/kex
    
    psi = np.power((R2a - R2b - pa*kex + pb*kex),2) - np.square(dw) + 4*pa*pb*np.square(kex)
    zeta = 2.*dw*(R2a - R2b - pa*kex + pb*kex)
    
    #Dplus_top = psi + 2.*np.square(dw)
    #Dplus_bot = np.sqrt(np.square(psi) + np.square(zeta))
    
    #Dplus = (1/2.)*(1 + Dplus_top/Dplus_bot)
    #Dminus = (1/2.)*(-1 + Dplus_top/Dplus_bot)
    
    Dplus = (1/2.)*(1 + ((psi + 2.*np.square(dw))/np.sqrt(np.square(psi) + np.square(zeta))))
    Dminus = (1/2.)*(-1 + ((psi + 2.*np.square(dw))/np.sqrt(np.square(psi) + np.square(zeta))))
    
    etaplus = (tcp/np.sqrt(2.))*np.sqrt(psi + np.sqrt(np.square(psi) + np.square(zeta)))
    etaminus = (tcp/np.sqrt(2.))*np.sqrt(-psi + np.sqrt(np.square(psi) + np.square(zeta)))
    
    #Dplus=0.5*(1+(psi+2*np.power(dw,2))/np.sqrt((np.power(psi,2)+np.power(zeta,2))))
    #Dminus=0.5*(-1+(psi+2*np.power(dw,2))/np.sqrt((np.power(psi,2)+np.power(zeta,2))))
    
    #etaplus=tcp/np.sqrt(2)*np.sqrt(psi+np.sqrt(np.power(psi,2)+np.power(zeta,2)))
    #etaminus=tcp/np.sqrt(2)*np.sqrt(-1*psi+np.sqrt(np.power(psi,2)+np.power(zeta,2)))
    
    #R2eff = R2a+0.5*(kex-1/tcp*np.arccosh(Dplus*np.cosh(etaplus)-Dminus*np.cos(etaminus)))
    
    R2eff = (1/2.)*(R2a + R2b + kex - (1/tcp)*(np.arccosh(Dplus*np.cosh(etaplus) - Dminus*np.cos(etaminus))))
    
    return R2eff

def PlotR2eff(vcpmg, R2eff):
    
    label_params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 2

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(vcpmg,R2eff,'ko')
    ax1.tick_params(direction='in',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=True,left=True,right=True)
    ax1.yaxis.major.formatter._useMathText = True
    ax1.set_xlabel("$\\nu_{CPMG}$ Hz",fontsize=14)
    ax1.set_ylabel('$R_{2,eff}$',fontsize=14)
    
    plt.show()
    
main()
    
    
    
    
    
    
