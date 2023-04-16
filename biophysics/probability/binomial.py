#!/usr/bin/env python3

import numpy as np
from scipy.special import comb
import pandas as pd

def total(n, x): # For getting probabilities of every state in the binomial probability distribution

    probs = {'n':[],'k':[],'x':[],'P':[]}
    for k in range(n+1):
        probs = iterate(probs, n, k, x)

    return pd.DataFrame(probs)

def states(n, k, x): # For getting probabilities of individual states in the binomial probability distribution

    probs = {'n':[],'k':[],'x':[],'P':[]}
    probs = iterate(probs, n, k, x)

    return pd.DataFrame(probs)

def iterate(probs, n, k, x):

    for xval in x:
        probs['n'].append(n)
        probs['k'].append(k)
        probs['x'].append(xval)
        probs['P'].append(comb(n,k)*(xval**k)*((1-xval)**(n-k)))

    return probs