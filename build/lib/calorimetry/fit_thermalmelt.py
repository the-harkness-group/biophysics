#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:37:13 2021

@author: robertharkness
"""

import sys
import pandas as pd
import numpy as np
import copy
from lmfit import Model
import matplotlib.pyplot as plt

def main():

    # Get data
    data = pd.read_csv(sys.argv[1])
    xtype = sys.argv[2]
    if xtype == 'Temperature':
        data[xtype] = data[xtype] + 273.15
    ytype = sys.argv[3]

    # Model
    gmod = Model(thermalmelt)

    # Fit
    result = gmod.fit(data[ytype],x=data[xtype],
    a=data[ytype].values[0],b=0,c=data[ytype].values[-1],d=0,e=-30,f=-0.004)

    # Simulate
    xsim = np.linspace(data[xtype].values[0],data[xtype].values[-1],100)
    thermal = thermalmelt(xsim, result.params['a'], result.params['b'], result.params['c'],
    result.params['d'], result.params['e'].value, result.params['f'].value)
    Plist = populations(xsim,result.params['e'].value, result.params['f'].value)

    # Plot
    plt.style.use('figure')
    fig,ax = plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(data[xtype]-273.15,data[ytype],'ko')
    ax[0].plot(xsim-273.15,thermal,'r--')
    ax[0].set_ylabel(ytype)
    ax[0].set_xlabel(xtype)
    ax[1].plot(xsim-273.15,Plist[0],'b')
    ax[1].plot(xsim-273.15,Plist[1],'g')
    ax[1].set_ylabel('Fraction')
    ax[1].set_xlabel(xtype)
    fig.tight_layout()
    fig.savefig('fit_result.pdf',format='pdf')
    print(result.fit_report())

# Two-state fit
def thermalmelt(x,a,b,c,d,e,f):
    
    R = 8.3145e-3
    dGF = e - x*f
    KF = np.exp(-dGF/(R*x))
    Q = 1 + KF
    PF = KF/Q
    PU = 1/Q
    
    f_bl = a + b*(x - x[0])
    u_bl = c + d*(x - x[0])

    thermal = PF*f_bl + PU*u_bl
        
    return thermal

# Two-state populations
def populations(x,e,f):

    R = 8.3145e-3
    dGF = e - x*f
    KF = np.exp(-dGF/(R*x))
    Q = 1 + KF
    PF = KF/Q
    PU = 1/Q

    return [PF, PU]

main()