#!/usr/bin/env python3

import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

def func(x,I0,D):
    """ x is gradient strength squared """
    gamma = 2.67513e4 # rads-1 G-1 (267.513e6 rads-1 T-1 * 1e-4 (Conversion to Gauss))
    Delta = 0.2 # diffusion time in s
    delta = 2*0.0014 #pulse duration in s
    return I0*np.exp(-1.*D*x*np.square(gamma*delta)*(Delta-delta/3.))

def main():

    """ Read data """
    params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(params['Dataset'])

    """ Gradient strengths """
    #### for gradients on bruker magnets
    gs = np.array(params['Gradient percentages']) # Difframp list of gradient attenuations
    g2s = np.square(gs*44.6) # Bruker 800

    #### for bruker 600 (uoft600)#########
    ####    gs = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) # Difframp list of gradient attenuations
    ####    g2s = np.square(gs*45)

    #### for bruker 800 (uoft800) #########
    ####    gs = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) # Difframp list of gradient attenuations
    ####    g2s = np.square(gs*44.6)

    #### for varian #########
    #    procpar = ng.varian.read_procpar("procpar")
    #    gs = np.array(procpar["gzlvl5"]["values"],dtype="float")
    #    print(gs)
        ### 0.001725 gauss/cmDAC for Curie ###
    #    g2s = np.square(gs*0.001725)

    groups = data.groupby('Assignment')
    pdf = PdfPages('Diffusion_2D_fits.pdf')
    for ind, group in groups:
        
        """ Fitting """
        """ Start params """
        I0 = group.Intensity.iloc[0] # Maximum signal
        D = 1e-10 # Diffusion constant

        """ Normalize intensities """
        I = group.Intensity.values/group.Intensity.iloc[0]

        popt, pcov = curve_fit(func, g2s, I, [I0,D])
        diag = np.diag(pcov)
        perr = np.sqrt(diag)
        result = "Fitting params\n"+r"I$_0$ = %8.3f"%popt[0]+"\n"+r"D = %8.3e $cm^2s^{-1}$"%popt[1] +"\n"+"+/-"+r"%8.3e $cm^2s^{-1}$"%(perr[1])
        print(result)

        """ Plotting fits """
        x = np.linspace(g2s.min(),g2s.max())
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        ax.plot(x,func(x,*popt),"k--")
        ax.plot(g2s,I,"o")
        ax.set_ylabel(r"I/I$_{0}$")
        ax.set_xlabel(r"G$^2$ $(G^2cm^{-2})$")
        ax.set_title(ind)
        ax.text(g2s.max()*.7,0.9,result)
        fig.tight_layout()
        pdf.savefig(fig)
    pdf.close()

if __name__=="__main__":
    main()