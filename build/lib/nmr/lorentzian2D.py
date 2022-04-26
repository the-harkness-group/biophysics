#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:03:21 2020

@author: robertharkness
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    
    # Set 2D peak parameters
    params = {}
    params['R2i'] = 8
    params['R2d'] = 12
    params['ampi'] = 1e6
    params['ampd'] = 1e6
    params['shifti'] = np.array([123.2,123.0,122.2,122.1])
    params['shiftd'] = np.array([8.8,9.5,8.4,8.3])
    params['starti'] = 120
    params['endi'] = 124
    params['startd'] = 8
    params['endd'] = 10
    params['contour_start'] = 0.002
    params['contour_num'] = 10
    params['contour_factor'] = 1.2
    params['npoints'] = 1e3
    
    # Make peak
    params = population_avgs(params)
    peak_dict = lorentzian2D(params)
    
    # Plot peak
    plot_lorentzian(params, peak_dict)


def population_avgs(params):
    
    # + coop
    P2 = np.array([1.00000000e+00, 9.75295249e-01, 9.42273651e-01, 8.67256973e-01, 7.04434951e-01, 3.94054571e-01, 7.38592657e-02, 7.74635204e-03, 1.07199029e-03, 1.72085224e-04])
    P2L = np.array([1.39265169e-19, 2.33117432e-02, 5.08626162e-02, 1.02473074e-01, 1.80254232e-01, 2.40126990e-01, 1.51299931e-01, 5.39208897e-02, 2.04830031e-02, 8.26157167e-03])
    P2L2 = np.array([5.51177114e-23, 1.39300733e-03, 6.86373255e-03, 3.02699526e-02, 1.15310817e-01, 3.65818439e-01, 7.74840803e-01, 9.38332758e-01, 9.78445007e-01, 9.91566343e-01])
    
    # - coop
    P2 = np.array([1.        , 0.97531226, 0.94246892, 0.86921682, 0.72037391, 0.48148682, 0.24072756, 0.09637431, 0.0326845 , 0.00924288])
    P2L = np.array([1.42709754e-19, 2.46721383e-02, 5.74435511e-02, 1.30294906e-01, 2.76963962e-01, 5.05258104e-01, 7.07315816e-01, 7.55544763e-01, 6.47063453e-01, 4.47969625e-01])
    P2L2 = np.array([5.51137671e-23, 1.56030646e-05, 8.75297183e-05, 4.88277559e-04, 2.66212569e-03, 1.32550747e-02, 5.19566256e-02, 1.48080926e-01, 3.20252047e-01, 5.42787496e-01])
    
    params['avg_i'] = P2*params['shifti'][0] + 0.5*P2L*params['shifti'][1] + 0.5*P2L*params['shifti'][2] + P2L2*params['shifti'][3]
    params['avg_d'] = P2*params['shiftd'][0] + 0.5*P2L*params['shiftd'][1] + 0.5*P2L*params['shiftd'][2] + P2L2*params['shiftd'][3]
    
    return params
    

def lorentzian2D(params):
    
    R2i = params['R2i']
    R2d = params['R2d']
    shifti = params['avg_i']*81.09
    shiftd = params['avg_d']*800
    iamp = params['ampi']
    damp = params['ampd']
    starti = params['starti']
    endi = params['endi']
    startd = params['startd']
    endd = params['endd']
    contour_start = params['contour_start']
    contour_factor = params['contour_factor']
    contour_num = params['contour_num']
    npoints = params['npoints']
    
    x_ppm = np.linspace(startd,endd,npoints)
    y_ppm = np.linspace(starti,endi,npoints)
    x_ppm_grid,y_ppm_grid = np.meshgrid(x_ppm,y_ppm)
    x_Hz_grid = x_ppm_grid*800
    y_Hz_grid = y_ppm_grid*81.09
    
    peak_dict = {}
    peak_counter = 0
    for direct_shift,indirect_shift in zip(shiftd,shifti):
        direct_peak = (2./(np.pi*R2d))*(1./(1 + np.square((x_Hz_grid - direct_shift)/(R2d/2.))))
        indirect_peak = (2./(np.pi*R2i))*(1./(1 + np.square((y_Hz_grid - indirect_shift)/(R2i/2.))))
    #peak = (iamp/np.pi)*(R2i/(np.square(YY - centeri) + np.square(R2i)))*(damp/np.pi)*(R2d/(np.square(XX - centerd) + np.square(R2d)))
        peak = direct_peak*indirect_peak
        peak_dict[f"peak_{peak_counter}"] = peak
        #peak_dict[f"peak_{direct_shift}"] = direct_peak
        #peak_dict[f"peak_{indirect_shift}"] = indirect_peak
        peak_counter += 1
    
    cl = contour_start * contour_factor ** np.arange(contour_num)
    
    peak_dict['x_ppm'] = x_ppm
    peak_dict['y_ppm'] = y_ppm
    peak_dict['x_ppm_grid'] = x_ppm_grid
    peak_dict['y_ppm_grid'] = y_ppm_grid
    peak_dict['cl'] = cl
    
    return peak_dict


def plot_lorentzian(params, peak_dict):
    
    fig,ax = plt.subplots(1,figsize=(11,4))
    peak_counter = 0
    for shiftd,shifti in zip(params['avg_d'],params['avg_i']):
        ax.contour(peak_dict[f"peak_{peak_counter}"],peak_dict['cl'],extent=(params['startd'],params['endd'],params['starti'],params['endi']))
        peak_counter +=1
        print(peak_counter)
        
    ax.set_xlim(params['endd'],params['startd'])
    ax.set_ylim(params['endi'],params['starti'])
    ax.set_xlabel('$^{1}$H ppm')
    ax.set_ylabel('$^{15}$N ppm')
    plt.show()
    

main()