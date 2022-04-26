#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:16:04 2020

@author: robertharkness
"""

import sys
import yaml
import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt

def main():
    
    params = yaml.safe_load(open(sys.argv[1],'r'))

    plot1Ds(params)
    

def plot1Ds(params):
    
    files = params['files']
    xlim = params['xlim']
    sample = params['sample']
    
    #fig,ax = plt.subplots(111,figsize=(11,4))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for f in files:
        
        dic,data = ng.pipe.read(f)
        uc = ng.pipe.make_uc(dic,data)
        
        ax.plot(uc.ppm_scale(),data,label=f)
        ax.set_title(sample)
        ax.set_xlim(xlim)
        ax.set_xlabel('$^{1}H$ ppm')
        ax.set_ylabel('Intensity')
        ax.legend()
    
    plt.show()
    
main()
    