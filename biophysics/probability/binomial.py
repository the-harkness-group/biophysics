#!/usr/bin/env python3

from urllib.request import ProxyBasicAuthHandler
import numpy as np
from scipy.special import comb
import pandas as pd

def total(n, x): # For getting probabilities of every state in the binomial probability distribution

    probs = {'n':[],'k':[],'x':[],'P':[]}
    for k in range(n+1):
        for xval in x:
            probs['n'].append(n)
            probs['k'].append(k)
            probs['x'].append(xval)
            probs['P'].append(comb(n,k)*(xval**k)*((1-xval)**(n-k)))
    df = pd.DataFrame(probs)

    return df # Return pandas dataframe

def states(n, k, x): # For getting probabilities of individual states in the binomial probability distribution

    probs = {'n':[],'k':[],'x':[],'P':[]}
    for xval in x:
        probs['n'].append(n)
        probs['k'].append(k)
        probs['x'].append(xval)
        probs['P'].append(comb(n,k)*(xval**k)*((1-xval)**(n-k)))

    return probs # Return dictionary