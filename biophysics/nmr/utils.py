#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 16:15:46 2021

@author: robertharkness
"""

import nmrglue as ng
import numpy as np

def read_data(file, dims=2, data_type='sparky'):
    "For importing NMR into a python useable format with nmrglue"

    if 'sparky' in data_type:
        dic, data = ng.sparky.read(file) # Spectrum

    for dim in range(dims): # Make ppm scales

        uc = ng.sparky.make_uc(dic, data, dim=dim)
        dic[f'ppm_scale_{dim}'] = uc.ppm_scale()
        dic[f'ppm_limits_{dim}'] = uc.ppm_limits()

    return dic, data

def get_contours(contour_start, contour_factor, num_contours):
    "Get parameters for plotting peak contours in NMR spectra"

    contour_dic = {}
    contour_dic['contour_start'] = contour_start
    contour_dic['contour_factor'] = contour_factor
    contour_dic['num_contours'] = num_contours
    contour_dic['contours'] = float(contour_start) * float(contour_factor) ** np.arange(float(num_contours))
    
    return contour_dic

def get_peak_box(data, dic, x, y, x_ppm_scale, y_ppm_scale, x_pad, y_pad):
    "Get subregion of NMR spectrum around a peak of interest, for example from a 2D spectrum or a 2D projection from a 3D spectrum"

    peak_box_dic = {}
    peak_box_dic['x_lim_indices'] = [(np.abs(x_ppm_scale - (x - x_pad))).argmin(), (np.abs(x_ppm_scale - (x + x_pad))).argmin()]
    peak_box_dic['y_lim_indices'] = [(np.abs(y_ppm_scale - (y - y_pad))).argmin(), (np.abs(y_ppm_scale - (y + y_pad))).argmin()]
    peak_box_dic['extent'] = (x_ppm_scale[peak_box_dic['x_lim_indices'][-1]], x_ppm_scale[peak_box_dic['x_lim_indices'][0]], 
                           y_ppm_scale[peak_box_dic['y_lim_indices'][-1]], y_ppm_scale[peak_box_dic['y_lim_indices'][0]])
    peak_box_dic['peak_box'] = data[peak_box_dic['y_lim_indices'][-1]:peak_box_dic['y_lim_indices'][0]+1,
        peak_box_dic['x_lim_indices'][-1]:peak_box_dic['x_lim_indices'][0]+1]
    
    return peak_box_dic

def reference_ppm_scale(dic, x_ref, y_ref, x_center, y_center, x_dim=0, y_dim=1):
    "Reference chemical shift scale for overlaying NMR spectra"

    dic[f'ppm_scale_{x_dim}_referenced'] = dic[f'ppm_scale_{x_dim}'] + (x_ref - x_center)
    dic[f'ppm_scale_{y_dim}_referenced'] = dic[f'ppm_scale_{y_dim}'] + (y_ref - y_center)

    return dic