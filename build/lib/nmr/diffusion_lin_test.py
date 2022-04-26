#!/usr/bin/env python3

import nmrglue as ng
import pylab as pl
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

def normalise(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def get_region(data,ppm_scale,start,end):
    region = data[:,start:end+1]
    region_ppm = ppm_scale[start:end+1]
    return region,region_ppm

def integrate(region):
    """ Data is 1d array """
    area = region.sum()
    return area

def func(x,I0,D):
    """ x is gradient strength squared """
    gamma = 2.67513e4 # rads-1 G-1 (267.513e6 rads-1 T-1 * 1e-4 (Conversion to Gauss))
    Delta = 0.15 # diffusion time in s
    delta = 2*0.000408263 #pulse duration in s
    return I0*np.exp(-1.*D*x*np.square(gamma*delta)*(Delta-delta/3.))

""" FUNCTION FOR FITTING GRADIENT CALIBRATION CONSTANT """
def calibration_func(P,I0,alpha):
    """ x is gradient strength squared """
    gamma = 2.67513e4 # rads-1 G-1 (267.513e6 rads-1 T-1 * 1e-4 (Conversion to Gauss))
    Delta = 0.15 # diffusion time in s
    delta = 2*0.000408263 # pulse duration in s
    """ FROM LEWIS' WATER SLED DIFFUSION PULSE SEQUENCE """
    """ D is diffusion HDO in D2O @25C, in Gd-doped sample """
    """ ; diffusion constant of HDO in D2O is 1.902e-5 cm^2/s at 25oC """
    """ Mills J of phys chem vol 77, no 5, 1973 """
    """ Longsworth, the mutual diffusion of light and heavy 1 vol 64, 1960, p 1914 """
    """ he gets 1.922 at 94% D2O. It will be a bit lower at higher amounts of D2O """
    """ On 600 we get 44.8-45.2 g/cm for 100% on Z based on 1 sample with Gd (ranjiths sample). """
    D = 1.902e-5 # HDO in D2O 25C from calibration experiment
    G = P*alpha # P is fractional gradient strength, multiplied by calibration parameter alpha gives gradient strength in Gauss
    return I0*np.exp(-1.*D*np.square(G*gamma*delta)*(Delta-delta/3.))

def main():

    """ Read data """
    # read NMR files
    dic,data = ng.pipe.read("./test.ft")
    uc = ng.pipe.make_uc(dic, data)
    ppm_scale = uc.ppm_scale()
    start_ppm = 4.9
    end_ppm = 4.6
    start_noise_ppm = 6.5
    end_noise_ppm = 6
    start = uc(start_ppm,"ppm")
    end = uc(end_ppm,"ppm")
    start_noise = uc(start_noise_ppm,"ppm")
    end_noise = uc(end_noise_ppm,"ppm")
    regions,region_ppm = get_region(data,ppm_scale,start,end)
    noise_regions,noise_region_ppm = get_region(data,ppm_scale,start_noise,end_noise)

    """ Integration """
    areas = np.array([integrate(i) for i in regions])
    noise = np.array([integrate(i) for i in noise_regions])

    """ normalise integrals """
    I0_0 = areas[0]
    areas = areas/areas[0]
    f = open('normalized-areas.txt', 'w')
    f.writelines(["%s\n"%str(i) for i in areas])
    f.close()
    err_I_I0 = noise*areas/I0_0*np.sqrt(np.square(1/areas)+1)

    """ Gradient strengths """
    #### for gradient calibration on bruker magnets
    gs = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) # Difframp list of gradient attenuations
    g2s = np.square(gs*44.0743)

    #### for bruker 600 (uoft600)#########
    ####    gs = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) # Difframp list of gradient attenuations
    ####    g2s = np.square(gs*45)

    #### for bruker 800 (uoft800) #########
    ####    gs = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) # Difframp list of gradient attenuations
    ####    g2s = np.square(gs*45)

    #### for varian #########
    #    procpar = ng.varian.read_procpar("procpar")
    #    gs = np.array(procpar["gzlvl5"]["values"],dtype="float")
    #    print(gs)
        ### 0.001725 gauss/cmDAC for Curie ###
    #    g2s = np.square(gs*0.001725)

    calibration = 'n' # Fit calibration constant flag

    if calibration == 'n':
        
        """ Fitting """
        """ Start params """
        I0 = areas[0] # Maximum signal
        D = 1e-10 # Diffusion constant

        popt, pcov = curve_fit(func, g2s, areas,[I0,D])
        diag = np.diag(pcov)
        perr = np.sqrt(diag)
        result = "Fitting params\n"+r"I$_0$ = %8.3f"%popt[0]+"\n"+r"D = %8.3e $cm^2s^{-1}$"%popt[1] +"\n"+"+/-"+r"%8.3e $cm^2s^{-1}$"%(perr[1])

        """ Plotting fits """
        x = np.linspace(g2s.min(),g2s.max())
        fig = pl.figure(figsize=(16,8))
        ax1 = fig.add_subplot(121)
        ax1.plot(x,func(x,*popt),"k--")
        ax1.plot(g2s,areas,"o")
        ax1.text(g2s.max()*.7,0.8,result)
        ax1.set_ylabel(r"I/I$_{0}$")
        ax1.set_xlabel(r"G$^2$ $(G^2cm^{-2})$")

        """ Plotting spectra """
        ax2 = fig.add_subplot(122)
        ax2.set_xlim(max(region_ppm),min(region_ppm))
        [ax2.plot(region_ppm,region,label="%d"%g2) for g2,region in zip(g2s,normalise(regions))]
        ax2.set_xlabel("ppm")
        ax2.set_ylabel("Normalized Intensity")
        ax2.legend(title="$G^{2} cm^{-2}$")
        pl.savefig("diffusion_plot.pdf")

    if calibration == 'y':

        """ Calibration fitting """
        """ Start params """
        I0 = areas[0] # Maximum signal
        alpha = 45 # Calibration constant
        popt, pcov = curve_fit(calibration_func, gs, areas,[I0,alpha])
        diag = np.diag(pcov)
        perr = np.sqrt(diag)
        print(f"Fitted I0 is {popt[0]}")
        print(f"Fittd alpha is {popt[1]}")

        """ Plotting fits """
        gs_sim = np.linspace(gs.min(),gs.max(),100)
        fig = pl.figure(figsize=(16,8))
        ax1 = fig.add_subplot(121)
        ax1.plot(gs_sim,calibration_func(gs_sim,*popt),"k--")
        ax1.plot(gs,areas,"o")
        ax1.set_ylabel(r"I/I$_{0}$")
        ax1.set_xlabel(r"Percent gradient strength")

        """ Plotting spectra """
        ax2 = fig.add_subplot(122)
        ax2.set_xlim(max(region_ppm),min(region_ppm))
        [ax2.plot(region_ppm,region,label=f"{g}%") for g,region in zip(gs*100,normalise(regions))]
        ax2.set_xlabel("ppm")
        ax2.set_ylabel("Normalized intensity")
        ax2.legend(title="Percent gradient strength")
        pl.savefig("diffusion_plot.pdf")

if __name__=="__main__":
    main()