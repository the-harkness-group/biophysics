#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 17:59:58 2021

@author: harkness
"""

import sys
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
from lmfit import minimize, Parameters, report_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
import customcolormap
import time
import pickle
import hydrodynamics
import twopathways
import bindingmodels
import fitting
import copy

### Read in data and set up for fitting
def main():

    params = yaml.safe_load(open(sys.argv[1],'r'))
    data = pd.read_csv(params['Dataset']) # Get dataset name to fit
    fit_data = data[data.Sample == params['Sample']] # Extract sample data to fit from dataset in cases where data might be for multiple samples
    fit_data = fit_data[fit_data.Temperature <= 52] # Select temperature range, DLS plate reader sometimes goes slightly above 50C e.g. 50.1 so extend a little beyond 50 here
    fit_data = fit_data[fit_data.Concentration.isin(params['Fit params']['Concentrations'])] # Select fit concentrations
    fit_data.Concentration = fit_data.Concentration*1e-6
    fit_data.Temperature = fit_data.Temperature + 273.15
    
    #fit_data = copy.deepcopy(fit_data.iloc[::3])

    fit_params = Parameters() # Set up fit/simulation parameters
    fit_params.add('dH1o', value=params['Fit params']['dH1o']['Value'],vary=params['Fit params']['dH1o']['Vary']) # dH1 at ref temp.
    fit_params.add('K1o', value=params['Fit params']['K1o']['Value'],vary=params['Fit params']['K1o']['Vary'],min=0) # K1 at ref temp.
    fit_params.add('dCp1', value=params['Fit params']['dCp1']['Value'],vary=params['Fit params']['dCp1']['Vary']) # K1 dCp
    fit_params.add('dH2o', value=params['Fit params']['dH2o']['Value'],vary=params['Fit params']['dH2o']['Vary']) # dH2 at ref temp.
    fit_params.add('K2o', value=params['Fit params']['K2o']['Value'],vary=params['Fit params']['K2o']['Vary'],min=0) # K2 at ref temp.
    fit_params.add('dCp2', value=params['Fit params']['dCp2']['Value'],vary=params['Fit params']['dCp2']['Vary']) # K2 dCp
    fit_constants = {'To':params['Fit params']['To']['Value']+273.15,'Rh_3':params['Fit params']['Rh_3']['Value'],
                     'eta coefficients':[params['Eta params'][0],params['Eta params'][1],params['Eta params'][2],
                     params['Eta params'][3]],'j_scale':params['Fit params']['j_scale'],'h_scale':params['Fit params']['h_scale'],
                    'N':params['Fit params']['N']['Value']}
    
    run_fit = params['Run fit'] # Run fit with 'y', simulate data with 'n'
    get_errors = params['Get errors'] # Do MC errors with 'y', bypass with 'n'
    MC_iter = params['Monte Carlo iterations'] # Number of Monte Carlo error iterations
 
    state_list = [f"{3*(x-1)}B" for x in range(1,fit_constants['N']+2)] # Make list of two pathway states up to final size
    state_list[0] = '3A'
    state_list[1] = '6A'
    state_list[2] = '6B'
    P_dict, C_dict = bindingmodels.make_dictionaries(state_list)
    D_dict = hydrodynamics.make_diffusion_dictionary(P_dict)
    
    #sim_func = twopathways.two_pathways
    sim_func = twopathways.two_pathways_roots
    observe_func = hydrodynamics.calculate_Dz
    wrapper_args = (sim_func,observe_func,fit_constants,P_dict,C_dict,D_dict)
    observe = 'D'
    #objective = fitting.objective
    objective = fitting.scaled_objective
    mc = fitting.MonteCarloErrors
    #mc = fitting.montecarlo_parallel
    
    print('You are using a scaled objective function!!!')

    if run_fit == 'y': # Fit data
        time0 = time.time() # Calculate fit run time
        result = minimize(objective, fit_params, method='nelder', args=(fit_data, 
        wrapper_func, wrapper_args, observe))
        time1 = time.time()
        print(f"\nThe elapsed fit time is {time1-time0}s \n")
        
        # Print optimized parameters, write dataframe
        report_fit(result)
        print('The reduced weighted RSS is:',(result.chisqr/(result.ndata-len(result.params.keys()))))
        opt_params = copy.deepcopy(result.params)
        save_dict = {'Parameter':[],'Value':[]}
        for k,v in opt_params.items():
            save_dict['Parameter'].append(k)
            save_dict['Value'].append(v.value)
        save_df = pd.DataFrame(save_dict)
        save_df.to_csv('optimal_fit_params.csv')

        if get_errors == 'y':
            time0 = time.time()
            
            # Calculate RMSD using fitting.objective since you need residuals! NOT SCALED RESIDUALS
            residuals = np.array(fitting.objective(opt_params, fit_data, wrapper_func, wrapper_args, observe)) # For RMSD
            RSS = np.sum(np.square(residuals))
            RMSD = np.sqrt(RSS/result.ndata)
            MC_dict, MC_errors = mc(fit_data, opt_params, fit_constants, wrapper_func,
            wrapper_args, observe, MC_iter, RMSD=RMSD, MC_objective=objective)
            time1 = time.time()
            print(f"\nThe total elapsed Monte-Carlo error iteration time is {time1-time0}s\n")

            plot_fit(fit_data, opt_params, wrapper_func, wrapper_args, fit_constants, params, MC_dict)

        if get_errors == 'n':
            plot_fit(fit_data, opt_params, wrapper_func, wrapper_args, fit_constants, params, MC_dict=None)

    if run_fit == 'n': # Simulate data, if optimal parameters already known
        if get_errors == 'y':
            
            # Calculate RMSD using fitting.objective since you need residuals! NOT SCALED RESIDUALS
            residuals = np.array(fitting.objective(fit_params, fit_data, wrapper_func, wrapper_args, observe)) # For RMSD
            RSS = np.sum(np.square(residuals))
            RMSD = np.sqrt(RSS/len(fit_data))

            time0 = time.time()
            MC_dict, MC_errors = mc(fit_data, fit_params, fit_constants, wrapper_func,
            wrapper_args, observe, MC_iter, RMSD=RMSD, MC_objective=objective)
            time1 = time.time()
            print(f"\nThe total elapsed Monte-Carlo error iteration time is {time1-time0}s\n")

            plot_fit(fit_data, fit_params, wrapper_func, wrapper_args, fit_constants, params, MC_dict)
            
        if get_errors == 'n':
            plot_fit(fit_data, fit_params, wrapper_func, wrapper_args, fit_constants, params, MC_dict=None)


# For getting observable quantity and passing to fit objective
def wrapper_func(fit_params, wrapper_args, temperature, concentration):
    
    sim_func, observe_func, fit_constants, P_dict, C_dict, D_dict = wrapper_args
    C_dict, P_dict = sim_func(fit_params, fit_constants, temperature, concentration,
    P_dict, C_dict)
    D_dict = observe_func(D_dict, C_dict, temperature, fit_constants['eta coefficients'], 
    fit_constants['Rh_3'])
    observable = D_dict['Dz']
    
    return observable


### Plot optimized fit result and save pdfs of plots for each temperature
def plot_fit(fit_data, opt_params, wrapper_func, wrapper_args, fit_constants, 
    params, MC_dict):
    
    # Set up plot parameters
    sim_func, observe_func, fit_constants, P_dict, C_dict, D_dict = wrapper_args
    pdf = matplotlib.backends.backend_pdf.PdfPages(params['Output PDF'])
    label_params = {'mathtext.default': 'regular' }
    plt.rcParams.update(label_params)
    plt.style.use('figure')
    hist_colors = ['#2166ac','#92c5de','#66c2a4','#d6604d','#b2182b',]
    hex_colors = ['#020101','#281112','#4E2128','#733243','#984364','#BD548A','#E166B4','#F374D4','#FF88F1','#FFA0FF','#FFBFFF']
    hex_colors = list(reversed(hex_colors))
    M3_color = '#fc8d59'
    M6_color = '#78c679'
    M24_color = '#2b8cbe'

    fit_groups = fit_data.groupby('Concentration')
    cmap = customcolormap.get_continuous_cmap(hex_colors,fit_groups.ngroups) #  Get HEX color at specified number of points using the above HEX colors
    
    figure_dir = params['Figure directory']
    if os.path.isdir(figure_dir) == False: # Check if directory for saving figures exists; if not, make it
        topdir = figure_dir.split('/')[0] + '/' + figure_dir.split('/')[1] # Figure path specified as ./Figures/Sample
        if os.path.isdir(topdir) == False:
            os.mkdir(topdir)
            os.mkdir(figure_dir) # Make bottom directory for the specific sample
        else:
            os.mkdir(figure_dir) # Make bottom directory for the specific sample
        
    # DLS data plots
    D_fig = plt.figure()
    D_ax = D_fig.add_subplot(111)
    D_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    D_ax.yaxis.major.formatter._useMathText = True
    D_ax.set_xlabel("Temperature \N{DEGREE SIGN}C")
    D_ax.set_ylabel('$\it{D_{z}}$ $cm^{2}$ $s^{-1}$')
    D_ax.set_yticks([2e-7,4e-7,6e-7,8e-7])
    D_axins = inset_axes(D_ax,width="37%",height="37%",loc=2,bbox_to_anchor=(0.03,0,1,1), bbox_transform=D_ax.transAxes)

    D_fig2, D_ax2 = plt.subplots(1,figsize=(3,5))
    D_ax2.yaxis.major.formatter._useMathText = True
    D_ax2.set_ylabel('$\it{D_{0}}$ $cm^{2}$ $s^{-1}$')
    D_ax2.bar([1],[2.33e-7+6.4e-10],width=0.48,color='k',label='experiment') # Experiment vs HYDROPRO diffusion constants for hexamer 95% confidence
    D_ax2.bar([1],[2.33e-7-6.4e-10],width=0.5,color=M6_color,label='experiment')
    D_ax2.bar([2],[2.35e-7],width=0.5,color=M6_color,label='HYDROPRO',hatch='///',edgecolor='w')
    D_ax2.bar([3],[2.70e-7+1.023e-9],width=0.48,color='k',label='experiment') # Experiment vs HYDROPRO diffusion constants for trimer 95% confidence
    D_ax2.bar([3],[2.70e-7-1.023e-9],width=0.5,color=M3_color,label='experiment')
    D_ax2.bar([4],[2.75e-7],width=0.5,color=M3_color,label='HYDROPRO',hatch='///',edgecolor='w')
    D_ax2.set_xticks([1,2,3,4])
    D_ax2.set_xticklabels(['$M_{6,expt.}$','$M_{6,calc.}$','$M_{3,expt.}$','$M_{3,calc.}$'],rotation=90,fontsize=36)
    
    max_temperature = 50 + 273.15
    for cidx, (ind, group) in enumerate(fit_groups):
        
        if (np.max(group.Temperature) < max_temperature):
            sim_dict = {'Temperature':np.linspace(np.min(group.Temperature),max_temperature,100),
                    'Concentration':np.full(100,group.Concentration.iloc[0])}
        
        else:
            sim_dict = {'Temperature':np.linspace(np.min(group.Temperature),np.max(group.Temperature),100),
                        'Concentration':np.full(100,group.Concentration.iloc[0])}
        
        sim_df = pd.DataFrame(sim_dict)
        Dz = np.zeros(len(sim_df.Temperature))
        
        for y, temperature in enumerate(sim_df.Temperature):
            Dz[y] = wrapper_func(opt_params, wrapper_args, temperature, sim_df.Concentration.iloc[y])

        D_ax.plot(group.Temperature-273.15,group.D*1e4,'o',color=cmap[cidx])
        D_ax.plot(sim_df['Temperature']-273.15,Dz*1e4,linewidth=2,color=cmap[cidx])
        
    DLS_Temperature = np.linspace(5,50,100) + 273.15
    D3 = np.zeros(len(DLS_Temperature))
    D6 = np.zeros(len(DLS_Temperature))
    D24 = np.zeros(len(DLS_Temperature))
    for y, temperature in enumerate(DLS_Temperature):
        
        eta = hydrodynamics.viscosity(temperature, fit_constants['eta coefficients'])
        D3[y] = hydrodynamics.stokes_diffusion(temperature, eta, fit_constants['Rh_3'])
        D6[y] = hydrodynamics.scaled_diffusion(D3[y],2,-0.227)
        D24[y] = hydrodynamics.scaled_diffusion(D3[y],8,-0.333)
        
    D_ax.plot(DLS_Temperature-273.15,D3*1e4,'--',linewidth=2,color=M3_color)
    D_ax.plot(DLS_Temperature-273.15,D6*1e4,'--',linewidth=2,color=M6_color)
    D_ax.plot(DLS_Temperature-273.15,D24*1e4,'--',linewidth=2,color=M24_color)

    # Diffusion constant inset at high temperature
    # Multiplying by 1e11 to make scale on inset match the full plot
    min_df = fit_data[fit_data['Concentration'] == min(fit_data.Concentration)]
    sim_dict = {'Temperature':np.linspace(np.min(min_df.Temperature),np.max(min_df.Temperature),100),
                    'Concentration':np.full(100,min_df.Concentration.iloc[0])}
    sim_df = pd.DataFrame(sim_dict)
    Dz = np.zeros(len(sim_df.Temperature))
    for y, temperature in enumerate(sim_df.Temperature):
        Dz[y] = wrapper_func(opt_params, wrapper_args, temperature, sim_df.Concentration.iloc[y])
    
    D_axins.plot(min_df.Temperature-273.15,min_df.D*1e11,'o',markersize=6,color=cmap[0])
    D_axins.plot(sim_df['Temperature']-273.15,Dz*1e11,linewidth=2,color=cmap[0])
    D_axins.plot(DLS_Temperature-273.15,D3*1e11,'--',linewidth=2,color=M3_color)
    D_axins.plot(DLS_Temperature-273.15,D6*1e11,'--',linewidth=2,color=M6_color)
    D_axins.plot(DLS_Temperature-273.15,D24*1e11,'--',linewidth=2,color=M24_color)
    D_axins.set_xticks([30, 40, 50])
    D_axins.set_yticks([5,6,7,8])
    D_axins.set_xlim([29.5,50.5])
    D_axins.set_ylim([4.8,8.5])
    D_axins.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    D_axins.yaxis.major.formatter._useMathText = True
    
    D_fig.tight_layout(pad=1)
    D_fig2.tight_layout(pad=1)
    pdf.savefig(D_fig)
    pdf.savefig(D_fig2)
    D_fig.savefig(f"{figure_dir}/full_Dz.pdf",format='pdf')
    D_fig2.savefig(f"{figure_dir}/Dbar.pdf",format='pdf')
    plt.close(D_fig)
    plt.close(D_fig2)
    
    # Plot equilibrium constants
    K_fig = plt.figure()
    K_ax = K_fig.add_subplot(111)
    K_ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
    K_ax.yaxis.major.formatter._useMathText = True
    K_ax.xaxis.major.formatter._useMathText = True
    K_ax.set_xlabel("1/T $K^{-1}$")
    K_ax.set_ylabel('ln($\it{K}$)')
    
    K_dict = {'K1':np.zeros(len(DLS_Temperature)),
    'K2':np.zeros(len(DLS_Temperature))}
    confidence_lnKs = {'lnK1':[],
    'lnK2':[]}
    if MC_dict: # Get MC Ks if required
        for x, element in enumerate(MC_dict['dH1o']):            
            for y, temperature in enumerate(DLS_Temperature):
                K_dict['K1'][y] = bindingmodels.equilibriumconstants(MC_dict['K1o'][x].value,
                MC_dict['dH1o'][x].value, MC_dict['dCp1'][x].value, fit_constants['To'],
                temperature)
                K_dict['K2'][y] = bindingmodels.equilibriumconstants(MC_dict['K2o'][x].value,
                MC_dict['dH2o'][x].value, MC_dict['dCp2'][x].value, fit_constants['To'],
                temperature)
            confidence_lnKs['lnK1'].append(np.log(K_dict['K1']))
            confidence_lnKs['lnK2'].append(np.log(K_dict['K2']))
                
    for y, temperature in enumerate(DLS_Temperature): # Get optimal Ks
        K_dict['K1'][y] = bindingmodels.equilibriumconstants(opt_params['K1o'].value,
        opt_params['dH1o'].value, opt_params['dCp1'].value, fit_constants['To'],
        temperature)
        K_dict['K2'][y] = bindingmodels.equilibriumconstants(opt_params['K2o'].value,
        opt_params['dH2o'].value, opt_params['dCp2'].value, fit_constants['To'],
        temperature)
    
    if not MC_dict: # No Monte Carlo errors
        K_ax.plot(1./(DLS_Temperature),np.log(K_dict['K1']),linewidth=2,color='#4393c3',label='$\it{K_{A}}$')
        K_ax.plot(1./(DLS_Temperature),np.log(K_dict['K2']),linewidth=2,color='#b2182b',label='$\it{K_{B}}$')

    if MC_dict: # Plot 95% Monte Carlo confidence intervals

        lnK1_upper = np.mean(confidence_lnKs['lnK1'],0) + 1.96*np.std(confidence_lnKs['lnK1'],0)
        lnK1_lower = np.mean(confidence_lnKs['lnK1'],0) - 1.96*np.std(confidence_lnKs['lnK1'],0)
        lnK2_upper = np.mean(confidence_lnKs['lnK2'],0) + 1.96*np.std(confidence_lnKs['lnK2'],0)
        lnK2_lower = np.mean(confidence_lnKs['lnK2'],0) - 1.96*np.std(confidence_lnKs['lnK2'],0)
        
        lnK1_mean = np.mean(confidence_lnKs['lnK1'],0)
        lnK2_mean = np.mean(confidence_lnKs['lnK2'],0)

        K_ax.plot(1./(DLS_Temperature),lnK1_mean,linewidth=2,color='#4393c3',label='$\it{K_{A}}$')
        K_ax.fill_between(1./(DLS_Temperature),lnK1_lower,lnK1_upper,facecolor='#4393c3',edgecolor='#4393c3',alpha=0.1)
        K_ax.plot(1./(DLS_Temperature),lnK2_mean,linewidth=2,color='#b2182b',label='$\it{K_{B}}$')
        K_ax.fill_between(1./(DLS_Temperature),lnK2_lower,lnK2_upper,facecolor='#b2182b',edgecolor='#b2182b',alpha=0.1)

    Kax_leg = K_ax.legend(loc='upper left',frameon=False,handlelength=0,fontsize=36)

    for line, text in zip(Kax_leg.get_lines(), Kax_leg.get_texts()):
        text.set_color(line.get_color())
        
    K_fig.tight_layout(pad=1.5)
    pdf.savefig(K_fig)
    if not MC_dict:
        K_fig.savefig(f"{figure_dir}/VantHoffplot.pdf",format='pdf')
    if MC_dict:
        K_fig.savefig(f"{figure_dir}/VantHoffplot_confidenceintervals.pdf",format='pdf')
    plt.close(K_fig)
    
    with open(f"{figure_dir}/lnK_confidence_dict.p", 'wb') as fp: # Write to dictionary
        confidence_lnKs['Temperature'] = DLS_Temperature - 273.15
        pickle.dump(confidence_lnKs, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Simulate populations and concentrations at desired biological monomer concentrations specified in parameter file
    # Plot population histograms for these concentrations
    for concentration in params['Simulation concentration']:
        for x, temperature in enumerate(params['Simulation temperature']):
            
            C_dict, P_dict = sim_func(opt_params, fit_constants, temperature+273.15, concentration*1e-6, P_dict, C_dict)
            
            KA = bindingmodels.equilibriumconstants(opt_params['K1o'].value,
                opt_params['dH1o'].value, opt_params['dCp1'].value, fit_constants['To'],
                temperature+273.15)
            KB = bindingmodels.equilibriumconstants(opt_params['K2o'].value,
                opt_params['dH2o'].value, opt_params['dCp2'].value, fit_constants['To'],
                temperature+273.15)
            print('#######################################################################')
            print(f"Temperature is: {temperature}C")
            print(f"Concentration is: {concentration} uM")
            print(f"KA from 3A + 3A <-> 6A concentrations is {C_dict['6A']/C_dict['3A']**2}")
            print(f"KA from fit is {KA}")
            print(f"KB from 3A + 3A <-> 6B concentrations is {C_dict['6B']/C_dict['3A']**2}")
            print(f"KB from 6B + 3A <-> 9B concentrations is {C_dict['9B']/(C_dict['6B']*C_dict['3A'])}")
            print(f"KB from 9B + 3A <-> 12B concentrations is {C_dict['12B']/(C_dict['9B']*C_dict['3A'])}")
            print(f"KB from fit is {KB}")

            Pfig, Pax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1,3]})# Make split axes plot

            # zoom-in / limit the view to different portions of the data
            Pax[0].set_ylim(0.25, 1.0)  # outliers only
            Pax[1].set_ylim(0, 0.20)  # most of the data
            Pax[0].set_yticks([0.25, 1.0])

            # hide the spines between ax and ax2
            Pax[0].spines['bottom'].set_visible(False)
            Pax[1].spines['top'].set_visible(False)
            Pax[0].xaxis.tick_top()
            Pax[1].tick_params(labeltop=False)  # don't put tick labels at the top
            Pax[1].xaxis.tick_bottom()

            Pax[0].yaxis.major.formatter._useMathText = True
            Pax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            Pax[0].set_xlim([0,41])

            Pax[1].yaxis.major.formatter._useMathText = True
            Pax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            Pax[1].set_xlabel("Oligomer size")
            Pax[1].set_xlim([0,41])
            Pax[1].set_xticks([0, 10, 20, 30, 40])
            Pax[1].set_xticklabels([0, 10, 20, 30, 40])
            Pfig.subplots_adjust(left=0.25)
            Pax[1].set_ylabel('.',color='w')
            Pfig.text(0.025,0.79,'Population',fontsize=36,rotation='vertical',transform=Pfig.transFigure,va='top')
            
            Psum = 0
            
            for z in range(1,int(fit_constants['N'])+1):
                
                if z == 1: # Trimer
                    
                    Pax[0].bar(3*z,P_dict[f"{3*z}A"],color=hist_colors[x],edgecolor=hist_colors[x],width=1)
                    Pax[1].bar(3*z,P_dict[f"{3*z}A"],color=hist_colors[x],edgecolor=hist_colors[x],width=1)
                    
                    Psum += P_dict[f"{3*z}A"]

                if z == 2: # For hexamer A and B

                    Pax[0].bar(2.5*z,P_dict[f"{3*z}A"],color=hist_colors[x],edgecolor=hist_colors[x],width=1)
                    Pax[1].bar(2.5*z,P_dict[f"{3*z}A"],color=hist_colors[x],edgecolor=hist_colors[x],width=1)
                    Pax[0].bar(3*z,P_dict[f"{3*z}B"],color=hist_colors[x],edgecolor=hist_colors[x],width=1)
                    Pax[1].bar(3*z,P_dict[f"{3*z}B"],color=hist_colors[x],edgecolor=hist_colors[x],width=1)
                    
                    Psum += P_dict[f"{3*z}A"]
                    Psum += P_dict[f"{3*z}B"]

                if z > 2:  # All other sizes

                    Pax[0].bar(3*z,P_dict[f"{3*z}B"],color=hist_colors[x],edgecolor=hist_colors[x],width=1)
                    Pax[1].bar(3*z,P_dict[f"{3*z}B"],color=hist_colors[x],edgecolor=hist_colors[x],width=1)
                    
                    Psum += P_dict[f"{3*z}B"]
            
            print('Population sum is:',Psum)
            print('#######################################################################\n')

            Pax[0].text(0.9,0.6,'{}\n$\it{{M_{{T}}}}$ = {} $\mu$M\n{} \N{DEGREE SIGN}C'.format(params['Sample name'],concentration,temperature),
                 fontsize=36,va='top',ha='right',ma='left',transform=Pax[0].transAxes)
            
            Pfig.tight_layout(pad=1)
            pdf.savefig(Pfig)
            Pfig.savefig(f"{figure_dir}/Populations_{temperature}C_{concentration}uM.pdf",format='pdf')
            plt.close(Pfig)
    
            with open(f"{figure_dir}/Populations_{concentration*1e6}uM_{temperature+273.15}C.p", 'wb') as fp:
                save_dict = copy.deepcopy(P_dict)
                save_dict['Temperature'] = sim_df['Temperature']
                pickle.dump(save_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
                fp.close()
            
            with open(f"{figure_dir}/Concentrations_{concentration*1e6}uM_{temperature+273.15}C.p", 'wb') as fp:
                save_dict = copy.deepcopy(C_dict)
                save_dict['Temperature'] = sim_df['Temperature']
                pickle.dump(save_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
                fp.close()
            
    pdf.close()
    
#main()
if __name__ == "__main__":  # _ALWAYS_ needed.  See https://stackoverflow.com/questions/419163/what-does-if-name-main-do
    main()
