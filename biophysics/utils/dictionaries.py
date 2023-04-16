#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:00:26 2021

@author: robertharkness
"""

import numpy as np

def make_dictionary_with_empty_arrays(keys, array_type='list', array_size=(1,1)):

    if array_type is 'list':
        the_dict = {k:[] for k in keys}

    if array_type is 'numpy':
        if isinstance(array_size,tuple) is False:
            raise TypeError('Array size must be of type tuple')
        if all(isinstance(v,int) for v in array_size) is False:
            raise TypeError('Array size tuple must contain only integers')
        for v in array_size:
            if v < 0:
                raise ValueError('Array size tuple must contain positive integers')
        else:
            the_dict = {k:np.zeros(array_size) for k in keys}

    return the_dict