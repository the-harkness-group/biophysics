#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: robertharkness
"""

import numpy as np

# Simple single exponential decay
def exponential(t, k):

    return np.exp(k*t)

# Simple single exponential decay with scaling factor
def scaled_exponential(t, a, k):

    return a*np.exp(k*t)

# Simple single exponential decay with scaling factor and non-zero intercept
def scaled_offset_exponential(t, a, b, k):

    return a*np.exp(k*t) + b