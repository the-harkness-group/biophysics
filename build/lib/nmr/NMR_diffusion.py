#!/usr/bin/env python3

import sys
import yaml
import nmrglue as ng
import numpy as np
from lmfit import Model, report_fit

import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.backends.backend_pdf import PdfPages

def get_region(data,ppm_scale,start,end):
    region = data[:,start:end+1]
    region_ppm = ppm_scale[start:end+1]
    return region,region_ppm

def load_yaml(yaml_file):
    """ reads files containing YAML and converts to dictionary """
    f = open(yaml_file)
    y = f.read()
    return yaml.load(y)

def data(fname,params):
    
    dic,data = ng.pipe.read(fname)
    uc = ng.pipe.make_uc(dic,data)
    ppm_scale = uc.ppm_scale()
    start_ppm = params["start_ppm"]
    end_ppm = params["end_ppm"]
    start = uc(start_ppm,"ppm")
    end = uc(end_ppm,"ppm")
    regions,region_ppm = get_region(data,ppm_scale,start,end)
    areas = np.array([i.sum() for i in regions])
    I0 = areas[0]
    areas = areas/I0
    # update dict
    params["areas"] = areas
    params["regions"] = regions
    params["region_ppm"] = region_ppm
    return params


def get_params(yaml):

    # retrieve dataset type (varian or bruker)
    dtype = yaml.get("dtype")
    gradients = yaml.get("gradients")
    procpar = yaml.get("procpar","procpar")
    if dtype=="varian":
        procpar = ng.varian.read_procpar(procpar)
        gs = np.array(procpar[gradients]["values"],dtype="float")
        conversion = yaml.get("conversion",0.00179)
        g2s = np.square(gs*conversion)
        yaml["delta"] = float(procpar[yaml["delta"]]["values"][0])
        yaml["Delta"] = float(procpar[yaml["T_diff"]]["values"][0])

    elif dtype=="bruker":
        pass

    else: raise(TypeError,"I don't know this file format")

    # update params
    yaml["gs"]=gs
    yaml["g2s"]=g2s
    if yaml["bipolar"]:
        yaml["delta"] = yaml["delta"] * 2.
    # check if single double or triple quantum
    if yaml["type"] == "1Q":
        yaml["factor"] = 1.0
    elif yaml["type"] == "2Q":
        yaml["factor"] = 2.0
    elif yaml["type"] == "3Q":
        yaml["factor"] = 3.0

    return yaml

def plot(params,pdf,bi_flag):
    # plot results
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(params["g2s"],params["areas"],"ro")
    ax1.plot(params["g2s"],params["out"].best_fit,"k--")
    
    if bi_flag == 'y':
        D1 = params["out"].params["D1"].value
        D2 = params["out"].params["D2"].value
        D1err = params["out"].params["D1"].stderr
        D2err = params["out"].params["D2"].stderr
        delta = params["out"].params["delta"].value
        Delta = params["out"].params["Delta"].value
        ax1.text(0.3,0.95,"D1 = %.2e $\pm$ %.2e\nD2 = %.2e $\pm$ %.2e\n$\delta$=%.3e\n$\Delta$=%.3e"%(D1,D1err,D2,D2err,delta,Delta),transform=ax1.transAxes,va="top")
        colors = [viridis(i) for i in np.linspace(0,1,len(params["regions"]))]
        [ax2.plot(params["region_ppm"],y,color=c,label="%d"%g2) for y,c,g2 in zip(params["regions"],colors,params["g2s"])]
        ax2.legend(title="G$^2$")
        ax2.invert_xaxis()
        ax1.set_xlabel("G$^2$ - $G^2cm^{-2}$")
        ax2.set_xlabel("ppm")
        ax1.set_ylabel("I/I$_0$")
        ax2.set_ylabel("I")
        plt.tight_layout()
        plt.suptitle(params["title"])
        
    else:
        D = params["out"].params["D"].value
        Derr = params["out"].params["D"].stderr
        delta = params["out"].params["delta"].value
        Delta = params["out"].params["Delta"].value
        ax1.text(0.3,0.95,"D = %.2e $\pm$ %.2e\n$\delta$=%.3e\n$\Delta$=%.3e"%(D,Derr,delta,Delta),transform=ax1.transAxes,va="top")
        colors = [viridis(i) for i in np.linspace(0,1,len(params["regions"]))]
        [ax2.plot(params["region_ppm"],y,color=c,label="%d"%g2) for y,c,g2 in zip(params["regions"],colors,params["g2s"])]
        ax2.legend(title="G$^2$")
        ax2.invert_xaxis()
        ax1.set_xlabel("G$^2$ - $G^2cm^{-2}$")
        ax2.set_xlabel("ppm")
        ax1.set_ylabel("I/I$_0$")
        ax2.set_ylabel("I")
        plt.tight_layout()
        plt.suptitle(params["title"])
    
    pdf.savefig()
    plt.close()


def fit(mod,params,bi_flag):

    if bi_flag == 'y':
        mod_params = mod.make_params(I01=1.0,D1=1e-9,D2=5e-7)
    else:
        mod_params = mod.make_params(I0=1.0,D=1e-7)
    
    print(mod_params)
    # set fixed params
    print("Setting Delta and delta")
    mod_params["Delta"].value = params["Delta"]
    mod_params["Delta"].vary=False
    mod_params["delta"].value= params["delta"]
    mod_params["delta"].vary=False
    mod_params["factor"].value= params["factor"]
    mod_params["factor"].vary=False
    print(mod_params)
    out = mod.fit(params["areas"],mod_params,x=params["g2s"])
    print(report_fit(out.params))
    params["out"] = out
    return params

def diffusion(x,I0,D,Delta,delta,factor):
    """ x is gradient strength squared """
    # bipolar
    gamma = 2.67513e4 # rads-1 G-1 (267.513e6 rads-1 T-1 * 1e-4 (Conversion to Gauss))
    
    #I0*np.exp(-1.*D*x*np.square(factor*gamma*delta)*(Delta-delta/3.))
    app_decay = I0*np.exp(-1.*D*x*np.square(factor*gamma*delta)*(Delta-delta/3.))
    
    return app_decay

def bi_diffusion(x,I01,D1,D2,Delta,delta,factor):
    """ x is gradient strength squared """
    # bipolar
    gamma = 2.67513e4 # rads-1 G-1 (267.513e6 rads-1 T-1 * 1e-4 (Conversion to Gauss))
    
    #I0*np.exp(-1.*D*x*np.square(factor*gamma*delta)*(Delta-delta/3.))
        
    I02 = I01
    decay_1 = I01*np.exp(-1.*D1*x*np.square(factor*gamma*delta)*(Delta-delta/3.))
    decay_2 = I02*np.exp(-1.*D2*x*np.square(factor*gamma*delta)*(Delta-delta/3.))
    app_decay = decay_1 + decay_2
    
    return app_decay

if __name__=="__main__":

    param_dict = load_yaml(sys.argv[1])
    # output pdf
    with PdfPages('fit_result.pdf') as pdf:
        for k in param_dict.keys():
            params = param_dict[k]
            params["title"] = k
            params = get_params(params)
            params = data(params["filename"],params)
            bi_flag = params['bi_flag']
            if bi_flag == 'y':
                mod = Model(bi_diffusion)
                params = fit(mod,params,bi_flag)
                plot(params,pdf,bi_flag)
            else:
                mod = Model(diffusion)
                params = fit(mod,params,bi_flag)
                plot(params,pdf,bi_flag)

            with open('%s.out'%k,'w') as out:
                out.write("G2\tArea\n")
                for g,a in zip(params["g2s"],params["areas"]):
                    out.write("%8.3f\t%8.3f\n"%(g,a))
        
