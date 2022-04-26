#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 08:00:33 2020

@author: toyam
"""

import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt


#Theoritical function of Reff, Carver-Richard solution
# Assuming R2A=R2B
def function(dw,R2,kex,pB,vcpmg):
	tcp=1.0/(2.*vcpmg)
	w=dw*2*np.pi
	psi=np.power(kex,2)-np.power(w,2)
	zeta=-2*w*kex*(1-2*pB)
	Dplus=0.5*(1+(psi+2*np.power(w,2))/np.sqrt((np.power(psi,2)+np.power(zeta,2))))
	Dminus=0.5*(-1+(psi+2*np.power(w,2))/np.sqrt((np.power(psi,2)+np.power(zeta,2))))
	etaplus=tcp/np.sqrt(2)*np.sqrt(psi+np.sqrt(np.power(psi,2)+np.power(zeta,2)))
	etaminus=tcp/np.sqrt(2)*np.sqrt(-1*psi+np.sqrt(np.power(psi,2)+np.power(zeta,2)))
	return R2+0.5*(kex-1/tcp*np.arccosh(Dplus*np.cosh(etaplus)-Dminus*np.cos(etaminus)))

#spin parameter list
kex=500.
pB= 0.01
kAB =pB*kex
kBA =(1-pB)*kex
IA0 = 1-pB
IB0 = pB
wA = 0		#offset of spin A (Hz)
wB =200	#offset of spin B (Hz)
R2A =10
R2B =10

#Relaxation and exchange matrix
Rp = np.zeros((2,2),dtype=complex)
Rp[0,0]=2*np.pi*wA*1j-R2A-kAB
Rp[0,1]=kBA
Rp[1,0]=kAB
Rp[1,1]=2*np.pi*wB*1j-R2B-kBA

Rm = np.zeros((2,2),dtype=complex)
Rm[0,0]=-2*np.pi*wA*1j-R2A-kAB
Rm[0,1]=kBA
Rm[1,0]=kAB
Rm[1,1]=-2*np.pi*wB*1j-R2B-kBA


#Caluculate IzA and IzB after t
I0 = np.zeros((2,1),dtype=complex)
I0[0,0]=IA0
I0[1,0]=IB0

# CPMG parameter
T = 0.04

vcpmg = np.arange(50,1050,50)
tcpmg = 0.25/vcpmg
n = T/(tcpmg*4.)

# Propagation
Refflist=np.empty_like(vcpmg,dtype=float)

for i in range(len(vcpmg)):
    
    Ucpmg = np.eye(2,dtype=complex)
    
    for k in range(int(n[i])): 
        Ucpmg = np.dot(sp.linalg.expm(Rp*tcpmg[i]),Ucpmg)
        Ucpmg = np.dot(sp.linalg.expm(Rm*tcpmg[i]),Ucpmg)
        Ucpmg = np.dot(sp.linalg.expm(Rm*tcpmg[i]),Ucpmg)
        Ucpmg = np.dot(sp.linalg.expm(Rp*tcpmg[i]),Ucpmg)
    
    It = np.dot(Ucpmg,I0)
    Refflist[i] = -1./T*np.log(np.real((It[0,0]+It[1,0])/(I0[0,0]+I0[1,0])))
        
plt.plot(vcpmg,Refflist,label="Bloch-McConnell propagator")
plt.plot(vcpmg,function(wB-wA,(R2A+R2B)/2.,kex,pB,vcpmg),label="Carver-Richard solution")
plt.xlabel("vCPMG (Hz)")
plt.ylabel("R2,eff (s-1)")
plt.legend()
plt.show()
