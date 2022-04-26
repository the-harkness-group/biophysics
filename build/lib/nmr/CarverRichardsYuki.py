#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:34:17 2020

@author: robertharkness
"""

import numpy as np
import matplotlib.pyplot as plt

def function(w,R2,kex,vcpmg,p):
    
    tcp=0.5/vcpmg
    
    psi=np.power(kex,2)-np.power(w,2)
    zeta=-2*w*kex*(1-2*p)
    
    Dplus=0.5*(1+(psi+2*np.power(w,2))/np.sqrt((np.power(psi,2)+np.power(zeta,2))))
    Dminus=0.5*(-1+(psi+2*np.power(w,2))/np.sqrt((np.power(psi,2)+np.power(zeta,2))))
    
    etaplus=tcp/np.sqrt(2)*np.sqrt(psi+np.sqrt(np.power(psi,2)+np.power(zeta,2)))
    etaminus=tcp/np.sqrt(2)*np.sqrt(-1*psi+np.sqrt(np.power(psi,2)+np.power(zeta,2)))
    
    return R2+0.5*(kex-1/tcp*np.arccosh(Dplus*np.cosh(etaplus)-Dminus*np.cos(etaminus)))

vcpmg=np.arange(50,1000,5)

plt.plot(vcpmg,function(50*2*np.pi,10,2500,vcpmg,0.10))
plt.show()