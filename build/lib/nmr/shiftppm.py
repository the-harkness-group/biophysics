#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 13:33:52 2021

@author: robertharkness
"""

import sys
import pandas as pd
import numpy as np

# For shifting assignments from 1H-13C HSQCs for U-1H/15N/13C
# samples to 1H-13C HMQCs for U-2H ILVM samples (i.e. accounting for isotope shifts)

# This only produces approximately shifted assignments based on a single peak
# of each type, but will be exact for the chosen peak to shift
def shiftppm():
    
    peak_list = pd.read_csv(sys.argv[1],delim_whitespace=True)
    
    # DegP_33 PDZ1 263-357
    L272_U13C15N1H = (24.912, 0.670) # CD1-HD1
    V309_U13C15N1H = (21.546, 0.736) # CG1-HG1
    M280_U13C15N1H = (17.175, 2.061) # CE-HE
    I318_U13C15N1H = (11.664, 0.706) # CD1-HD1
    
    L272_ILVM = (24.669, 0.736) # CD1-HD1
    V309_ILVM = (21.415, 0.802) # CG1-HG1
    M280_ILVM = (17.266, 2.131) # CE-HE
    I318_ILVM = (11.382, 0.759) # CD1-HD1
    
    L272_shift = (L272_ILVM[0]-L272_U13C15N1H[0], L272_ILVM[1]-L272_U13C15N1H[1])
    V309_shift = (V309_ILVM[0]-V309_U13C15N1H[0], V309_ILVM[1]-V309_U13C15N1H[1])
    M280_shift = (M280_ILVM[0]-M280_U13C15N1H[0], M280_ILVM[1]-M280_U13C15N1H[1])
    I318_shift = (I318_ILVM[0]-I318_U13C15N1H[0], I318_ILVM[1]-I318_U13C15N1H[1])
    
    peak_list_ppm = peak_list.iloc[:, 0:3] # In case peak list has ppm and Hz, just take ppm part
    new_list = peak_list_ppm.copy()
    
    with open('P33_U13C15N1H_toILVM_shifted_assignments_50C.txt','w') as file:
        
        file.write('      Assignment         w1         w2\n\n') 
        
        for assignment in new_list.Assignment:
        
            if assignment[0] == 'L':
                
                shifted_assignment_13C = np.round(new_list.loc[new_list['Assignment'] == assignment, 'w1'].values[0] + L272_shift[0],3) #13C
                shifted_assignment_1H = np.round(new_list.loc[new_list['Assignment'] == assignment, 'w2'].values[0] + L272_shift[1],3) #1H
                file.write(f"      {assignment}     {shifted_assignment_13C}      {shifted_assignment_1H}\n")
                
            if assignment[0] == 'V':
            
                shifted_assignment_13C = np.round(new_list.loc[new_list['Assignment'] == assignment, 'w1'].values[0] + V309_shift[0],3)
                shifted_assignment_1H = np.round(new_list.loc[new_list['Assignment'] == assignment, 'w2'].values[0] + V309_shift[1],3)
                file.write(f"      {assignment}     {shifted_assignment_13C}      {shifted_assignment_1H}\n")
            
            if assignment[0] == 'M':
            
                shifted_assignment_13C = np.round(new_list.loc[new_list['Assignment'] == assignment, 'w1'].values[0] + M280_shift[0],3)
                shifted_assignment_1H = np.round(new_list.loc[new_list['Assignment'] == assignment, 'w2'].values[0] + M280_shift[1],3)
                file.write(f"        {assignment}     {shifted_assignment_13C}      {shifted_assignment_1H}\n")
            
            if assignment[0] == 'I':
            
                shifted_assignment_13C = np.round(new_list.loc[new_list['Assignment'] == assignment, 'w1'].values[0] + I318_shift[0],3)
                shifted_assignment_1H = np.round(new_list.loc[new_list['Assignment'] == assignment, 'w2'].values[0] + I318_shift[1],3)
                file.write(f"      {assignment}     {shifted_assignment_13C}      {shifted_assignment_1H}\n")
                
    file.close()
    
    #new_list.to_csv('P33_U13C15N1H_toILVM_shifted_assignments_50C.txt',sep=' ',mode='a')
    
shiftppm()