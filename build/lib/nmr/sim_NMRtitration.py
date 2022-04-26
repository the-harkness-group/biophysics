#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:52:24 2019

@author: harkness
"""
########################################################################################

# Calculate concentrations for an NMR titration using excess binding partner in one
# tube (B) and no binding partner in another tube (A)

# As the titration proceeds both A and B will contain binding partner

# The concentration of the protein does not change and only the concentration
# of the substrate/ligand binding partner is varied by mixing the two endpoints

# CB = concentration of the binding partner in the B tube
# CA = concentration of the binding partner in the A tube
# index i refers to the current step of the titration

# CB,i = CB,i-1 - (CB,i-1 - CA,i-1)(Vmix/Vo)
# CA,i = CA,i-1 + (CB,i-1 - CA,i-1)(Vmix/Vo)

# This coupled set of equations shows that the concentration of binding partner in
# the B tube initially decreases as a function of i, and the concentration of binding
# partner in the A tube initially increases as a function of i
# They they both meet at the same intermediate value of the binding partner as a 
# result of the mixing

######################################################################################

import numpy as np
import matplotlib.pyplot as plt

Vmix = [0,2,5,10,25,50]
Vo = 150

steps = len(Vmix)
PT = 100e-6

CB = np.zeros(steps)
CB[0] = 5000e-6
CA = np.zeros(steps)
CA[0] = 0

step_index = np.arange(1,len(Vmix))

for idx in step_index:
       CA[idx] = CA[idx-1] + (CB[idx-1] - CA[idx-1])*(Vmix[idx]/Vo)
       CB[idx] = CB[idx-1] + (CA[idx-1] - CB[idx-1])*(Vmix[idx]/Vo)
       
print('The concentration in tube A is: ',CA*1e6,'uM')
#print('The molar ratio in the A tube is: ',CA/PT,' uM')
print('The concentration in tube B is: ',CB*1e6,'uM')
#print('The molar ratio in the B tube is: ',CB/PT,' uM')

index = np.arange(1,steps+1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(index,CA,'co')
ax.plot(index,CB,'ro')
ax.set_xlabel('Titration step')
ax.set_ylabel('Molar ratio')

