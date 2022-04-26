#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: robertharkness
"""

import numpy as np

# Fit line to initial linear portion of kinetic data to get initial rates
def linear_fit(x, y):

    line_params = np.polyfit(x, y, 1)

    return line_params

# Simulate line using fitted initial rate parameters
def linear_sim(x, params):

    x = np.array(x) # ensure numpy array
    line = params[0]*x + params[1] # line is y = mx + b

    return line