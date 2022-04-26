#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:03:25 2020

@author: robertharkness
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from functools import partial
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
import matplotlib


#########################################################################
##################### MAIN FUNCTION #####################################
#########################################################################
def main():
    
    ##########################################################################
    ############################## PARAMETERS ################################
    ##########################################################################
    
    # Simulation parameters
    sim_params = {}
    sim_params['PT'] = 100e-6 # Titration concentrations and points
    sim_params['sim_points'] = 100
    #sim_params['LT'] = np.linspace(0,50*sim_params['PT'],sim_params['sim_points'])
    sim_params['LT'] = np.hstack([0,np.logspace(np.log10(0.05*sim_params['PT']),np.log10(50*sim_params['PT']),sim_params['sim_points']-1,endpoint=True)])
    sim_params['BF_d'] = 800
    sim_params['BF_i'] = 81.09
    sim_params['R2i'] = 6 # Relaxation rates
    sim_params['R2d'] = 20
    sim_params['starti'] = 122.75 # Spectral windows
    sim_params['endi'] = 123.15
    sim_params['startd'] = 8.55
    sim_params['endd'] = 8.85
    sim_params['contour_start'] = 0.003 # Contour plot params
    sim_params['contour_num'] = 3
    sim_params['contour_factor'] = 1.1
    sim_params['npoints'] = 500 # Spectrum points
    
    sim_params['wUU_dir'] = 8.7 # Chemical shifts in ppm for 1H and 15N
    sim_params['wUB_dir'] = 8.8
    sim_params['wBU_dir'] = 8.8
    sim_params['wBB_dir'] = 8.6
    sim_params['wUU_ind'] = 123.1 
    sim_params['wUB_ind'] = 123.0
    sim_params['wBU_ind'] = 122.9
    sim_params['wBB_ind'] = 122.8

    sim_params['intensity'] = 1000 # Slow exchange intensity
    sim_params['K'] = 5000 # 200 uM Kd
    sim_params['alphas'] = np.array([10,1,0.1]) # Positive, non, anti cooperative
    sim_params['Models'] = ['Fast','Slow'] # Fast or slow exchange
    spec_dict = spectral_params(sim_params)
    
    ##########################################################################
    ############################## SIMULATION ################################
    ##########################################################################
    
    # Iterate through models, cooperativities and simulate data for dimer model
    # Store results in dictionary for plotting at the end
    sim_data, Pop_dict = NMRtitration(sim_params, spec_dict)

    ##########################################################################
    ############################## PLOTTING ##################################
    ##########################################################################

    # Call plotting function with simulation/fit result
    plot_result(sim_data, sim_params, spec_dict)
        

# Calculate fast and slow exchange NMR titration observables
def NMRtitration(sim_params, spec_dict):
    
    # Unpack parameters for fitting
    LT = sim_params['LT']
    PT = sim_params['PT']
    K = sim_params['K']
    
    # Make dictionary for storing simulated data
    NMR_dict = {}
    for alpha in sim_params['alphas']:
        # Fast exchange
        NMR_dict[f"F_{alpha}"] = {}
        NMR_dict[f"F_{alpha}"]['Two_freqs'] = {}
        NMR_dict[f"F_{alpha}"]['Two_freqs']['dir'] = []
        NMR_dict[f"F_{alpha}"]['Two_freqs']['ind'] = []
        NMR_dict[f"F_{alpha}"]['Two_freqs']['CSP'] = np.zeros(len(sim_params['LT']))
        NMR_dict[f"F_{alpha}"]['Four_freqs'] = {}
        NMR_dict[f"F_{alpha}"]['Four_freqs']['dir'] = []
        NMR_dict[f"F_{alpha}"]['Four_freqs']['ind'] = []
        NMR_dict[f"F_{alpha}"]['Four_freqs']['CSP'] = np.zeros(len(sim_params['LT']))
        
        # Slow exchange
        NMR_dict[f"S_{alpha}"] = {}
        NMR_dict[f"S_{alpha}"]['U_app'] = np.zeros(len(sim_params['LT']))
        NMR_dict[f"S_{alpha}"]['B_app'] = np.zeros(len(sim_params['LT']))
    
    # Iterate through models and cooperativities, simulate NMR data
    for model in sim_params['Models']:
        
        for alpha in sim_params['alphas']:
    
            # Solve for unknown concentrations numerically using fsolve
            solutions = [] # List of solutions   
            for ligand in LT:
                p = [PT,1e-6,1e-6,ligand]
                q = [PT,ligand,alpha,K]
                ffs_partial = partial(ffs_complex,q)
                # Solutions are ordered according to how the initial guess vector is arranged
                solutions.append(opt.root(ffs_partial,p,method='lm'))
                
            # Calculate fast or slow exchange observables in titration
            Pop_dict = Populations(PT,solutions)
            plt.plot(LT,Pop_dict['P2'],'r',label='P2') # Plot populations to check that they work out and change as you expect according to the cooperativity
            plt.plot(LT,Pop_dict['P2L'],'b',label='P2L')
            plt.plot(LT,Pop_dict['P2L2'],'g',label='P2L2')
            plt.plot(LT,Pop_dict['U_app'],'y',label='U_app=P2+0.5P2L')
            plt.plot(LT,Pop_dict['B_app'],'m',label='B_app=0.5P2L+P2L2')
            plt.plot(LT,Pop_dict['P2']+Pop_dict['P2L']+Pop_dict['P2L2'],'k',label='P2+P2L+P2L2')
            plt.xlabel('LT M')
            plt.ylabel('Population')
            plt.legend()
            plt.show()
            if model == 'Fast':
        
                # Fast exchange
                # Convert ppm shifts to Hz with arbitrary spectrometer frequency for direct and indirect dimensions to calculate CSP
                wUU_dir = sim_params['wUU_dir'] * sim_params['BF_d']
                wUB_dir = sim_params['wUB_dir'] * sim_params['BF_d']
                wBU_dir = sim_params['wBU_dir'] * sim_params['BF_d']
                wBB_dir = sim_params['wBB_dir'] * sim_params['BF_d']
                
                wUU_ind = sim_params['wUU_ind'] * sim_params['BF_i']
                wUB_ind = sim_params['wUB_ind'] * sim_params['BF_i']
                wBU_ind = sim_params['wBU_ind'] * sim_params['BF_i']
                wBB_ind = sim_params['wBB_ind'] * sim_params['BF_i']
                
                # Dimer case with two or four observed frequencies
                NMR_dict[f"F_{alpha}"]['Two_freqs']['dir'] = Pop_dict['U_app']*wUU_dir + np.array(Pop_dict['B_app'])*wBB_dir
                NMR_dict[f"F_{alpha}"]['Two_freqs']['ind'] = Pop_dict['U_app']*wUU_ind + np.array(Pop_dict['B_app'])*wBB_ind
                NMR_dict[f"F_{alpha}"]['Four_freqs']['dir'] = Pop_dict['P2']*wUU_dir + 0.5*Pop_dict['P2L']*wUB_dir + 0.5*Pop_dict['P2L']*wBU_dir + Pop_dict['P2L2']*wBB_dir
                NMR_dict[f"F_{alpha}"]['Four_freqs']['ind'] = Pop_dict['P2']*wUU_ind + 0.5*Pop_dict['P2L']*wUB_ind + 0.5*Pop_dict['P2L']*wBU_ind + Pop_dict['P2L2']*wBB_ind
                
                # Calculate CSP relative to first titration point where LT/PT = 0, calculate 2D lorentzians at each titration point
                for index, element in enumerate(NMR_dict[f"F_{alpha}"]['Two_freqs']['dir']):
                    NMR_dict[f"F_{alpha}"]['Two_freqs']['CSP'][index] = np.sqrt( np.square(NMR_dict[f"F_{alpha}"]['Two_freqs']['dir'][index] - NMR_dict[f"F_{alpha}"]['Two_freqs']['dir'][0]) + np.square(NMR_dict[f"F_{alpha}"]['Two_freqs']['ind'][index] - NMR_dict[f"F_{alpha}"]['Two_freqs']['ind'][0]) )
                    NMR_dict[f"F_{alpha}"]['Four_freqs']['CSP'][index] = np.sqrt( np.square(NMR_dict[f"F_{alpha}"]['Four_freqs']['dir'][index] - NMR_dict[f"F_{alpha}"]['Four_freqs']['dir'][0]) + np.square(NMR_dict[f"F_{alpha}"]['Four_freqs']['ind'][index] - NMR_dict[f"F_{alpha}"]['Four_freqs']['ind'][0]) )
                
                    NMR_dict[f"F_{alpha}"]['Two_freqs'][f"peak_{index}"] = lorentzian2D(sim_params,spec_dict,NMR_dict[f"F_{alpha}"]['Two_freqs']['dir'][index],NMR_dict[f"F_{alpha}"]['Two_freqs']['ind'][index])
                    NMR_dict[f"F_{alpha}"]['Four_freqs'][f"peak_{index}"] = lorentzian2D(sim_params,spec_dict,NMR_dict[f"F_{alpha}"]['Four_freqs']['dir'][index],NMR_dict[f"F_{alpha}"]['Four_freqs']['ind'][index])
                    
            if model == 'Slow':
    
                # Slow exchange
                NMR_dict[f"S_{alpha}"]['U_app'] = np.array(Pop_dict['U_app'])*sim_params['intensity']
                NMR_dict[f"S_{alpha}"]['B_app'] = np.array(Pop_dict['B_app'])*sim_params['intensity']
    
    return NMR_dict, Pop_dict


# Equations for dimer binding to two ligands sequentially with cooperativity, P2 + L <-> P2L + L <-> P2L2
def ffs_complex(q,p):
    
    # Unpack variables and constants
    P2, P2L, P2L2, L = p # Variables
    PT, LT, alpha, K = q # Constants
    
    # Equations have to be set up as equal to 0
    eq1 = -PT + P2 + P2L + P2L2 # Protein equation
    eq2 = -LT + L + P2L + 2*P2L2 # Ligand equation
    eq3 = 2*K*P2*L - P2L
    eq4 = (1/2)*alpha*K*P2L*L - P2L2
    
    return [eq1, eq2, eq3, eq4]


# Calculate populations from the solver solutions
def Populations(PT, solutions):
    
    P2 = np.zeros(len(solutions))
    P2L = np.zeros(len(solutions))
    P2L2 = np.zeros(len(solutions))
    U_app = np.zeros(len(solutions))
    B_app = np.zeros(len(solutions))
        
    for index, sol in enumerate(solutions):
            
        P2[index] = sol.x[0]/PT # Unbound dimer
        P2L[index] = sol.x[1]/PT # 1-bound dimer
        P2L2[index] = sol.x[2]/PT # 2-bound dimer
        U_app[index] = (sol.x[0]  + 0.5*sol.x[1])/PT # Apparent unbound is sum of unbound protomers
        B_app[index] = (0.5*sol.x[1] + sol.x[2])/PT # Apparent bound is sum of bound protomers
            
    Pop_dict = {'P2':P2,'P2L':P2L,'P2L2':P2L2,'U_app':U_app,'B_app':B_app}

    return Pop_dict


# Get spectral plotting and simulation parameters
def spectral_params(sim_params):
    
    starti = sim_params['starti']
    endi = sim_params['endi']
    startd = sim_params['startd']
    endd = sim_params['endd']
    contour_start = sim_params['contour_start']
    contour_factor = sim_params['contour_factor']
    contour_num = sim_params['contour_num']
    npoints = sim_params['npoints']
    
    x_ppm = np.linspace(startd,endd,npoints)
    y_ppm = np.linspace(starti,endi,npoints)
    x_ppm_grid,y_ppm_grid = np.meshgrid(x_ppm,y_ppm)
    x_Hz_grid = x_ppm_grid*sim_params['BF_d']
    y_Hz_grid = y_ppm_grid*sim_params['BF_i']
    
    cl = contour_start * contour_factor ** np.arange(contour_num)
    
    spec_dict = {'x_ppm':x_ppm,'y_ppm':y_ppm,'x_Hz_grid':x_Hz_grid,'y_Hz_grid':y_Hz_grid,'cl':cl}
    
    return spec_dict


# Simulate 2D peak in Hz units
def lorentzian2D(sim_params, spec_dict, shiftd, shifti):

    direct_peak = (2./(np.pi*sim_params['R2d']))*(1./(1 + np.square((spec_dict['x_Hz_grid'] - shiftd)/(sim_params['R2d']/2.))))
    indirect_peak = (2./(np.pi*sim_params['R2i']))*(1./(1 + np.square((spec_dict['y_Hz_grid'] - shifti)/(sim_params['R2i']/2.))))
    peak = direct_peak*indirect_peak
    
    return peak


# Plot solver results and save pdf  
def plot_result(Plot_dict, sim_params, spec_dict):
    
    # Set up PDF and plotting options for nice plots
    label_params = {'mathtext.default': 'regular' }
    plt.rcParams.update(label_params)
    plt.rcParams['axes.linewidth'] = 4
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    fastcolors = ['#7a0177','#dd3497','#fa9fb5']
    slowcolors = ['#253494','#1d91c0','#7fcdbb']
    c_colors = cm.jet(np.linspace(0,1,len(sim_params['LT'][1:-1:9])+1))
    
    # Figure panels
    fig1, ax1 = plt.subplots(1,figsize=(11,6))
    fig2, ax2 = plt.subplots(1,figsize=(11,6))
    fig3, ax3 = plt.subplots(1,figsize=(11,6))
    fig4, ax4 = plt.subplots(1,figsize=(11,7))
    fig5, ax5 = plt.subplots(1,figsize=(11,7))
    fig6, ax6 = plt.subplots(1,figsize=(11,7))
    fig7, ax7 = plt.subplots(1,figsize=(11,7))
    ax_list = [ax1,ax2,ax3,ax4,ax5,ax6,ax7]
    
    # Insets
    axins = []    
    axins.append(inset_axes(ax4, width="45%", height="45%", loc=4, borderpad=10))
    axins.append(inset_axes(ax5, width="45%", height="45%", loc=4, borderpad=10))
    axins.append(inset_axes(ax6, width="45%", height="45%", loc=4, borderpad=10))
    
    for c, ax in enumerate(ax_list):
        ax.tick_params(direction='in',axis='both',length=7,width=5,grid_alpha=0.3,bottom=True,top=True,left=True,right=True,labelsize=36)
        ax.yaxis.major.formatter._useMathText = True
        ax.yaxis.get_offset_text().set_fontsize(36)
        ax.yaxis.get_offset_text().set_fontweight('bold')
        
        if 3 <= c <= 6:
            ax.grid(linestyle='dashed',linewidth=3,dash_capstyle='round',dashes=(1,3))

        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(36)
            item.set_fontweight('bold')
            
    for c, ax in enumerate(axins):
        ax.tick_params(direction='in',axis='both',length=6,width=4,grid_alpha=0.3,bottom=True,top=True,left=True,right=True,labelsize=36)
        ax.yaxis.major.formatter._useMathText = True
        ax.yaxis.get_offset_text().set_fontsize(36)
        ax.yaxis.get_offset_text().set_fontweight('bold')
        
        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(36)
            item.set_fontweight('bold')
            
    # Plot simulation results
    for index, alpha in enumerate(sim_params['alphas']):
        
        c_index=0
        
        if index == 0:
            for number,ligand in enumerate(sim_params['LT']):
                if number == 0 or number % 9 == 0:
                    ax4.plot(sim_params['LT']/sim_params['PT'],Plot_dict[f"F_{alpha}"]['Two_freqs']['CSP'],linewidth=5,color=fastcolors[index])
                    ax1.contourf(Plot_dict[f"F_{alpha}"]['Four_freqs'][f"peak_{number}"],spec_dict['cl'],extent=(sim_params['startd'],sim_params['endd'],sim_params['starti'],sim_params['endi']),linewidths=5,colors=[c_colors[c_index],c_colors[c_index],c_colors[c_index]])
                    axins[0].contourf(Plot_dict[f"F_{alpha}"]['Two_freqs'][f"peak_{number}"],spec_dict['cl'],extent=(sim_params['startd'],sim_params['endd'],sim_params['starti'],sim_params['endi']),linewidths=5,colors=[c_colors[c_index],c_colors[c_index],c_colors[c_index]])
                    c_index += 1
             
            ax1.text(0.05, 0.95,f"$\\it{{\\alpha}}$ = {alpha}",horizontalalignment='left', verticalalignment='top',transform=ax1.transAxes,fontsize=48,fontweight='bold')
            ax1.set_xlabel('$^{1}$H ppm',fontsize=48,fontweight='bold')
            ax1.set_ylabel('$^{15}$N ppm',fontsize=48,fontweight='bold')
            ax1.set_yticks([123.1,123.0,122.9,122.8])
            ax1.set_xlim(sim_params['endd'],sim_params['startd'])
            ax1.set_ylim(sim_params['endi'],sim_params['starti'])
            ax4.set_xlabel('$L_{T}$/$P_{T}$',fontsize=48,fontweight='bold')
            ax4.set_ylabel('CSP [Hz]',fontsize=48,fontweight='bold')
            ax4.text(0.6, 0.85,f"$\\it{{\\alpha}}$ = {alpha}",horizontalalignment='left', verticalalignment='top',transform=ax4.transAxes,fontsize=48,fontweight='bold',color=fastcolors[index])
            axins[0].set_xlabel('$^{1}$H ppm')
            axins[0].set_ylabel('$^{15}$N ppm')
            axins[0].set_yticks([123.1,122.8])
            axins[0].set_xlim(sim_params['wUU_dir']+0.02,sim_params['wBB_dir']-0.02)
            axins[0].set_ylim(sim_params['wUU_ind']+0.05,sim_params['wBB_ind']-0.05)
                    
        if index == 1:
            for number,ligand in enumerate(sim_params['LT']):
                if number == 0 or number % 9 == 0:
                    ax5.plot(sim_params['LT']/sim_params['PT'],Plot_dict[f"F_{alpha}"]['Two_freqs']['CSP'],linewidth=5,color=fastcolors[index])
                    ax2.contourf(Plot_dict[f"F_{alpha}"]['Four_freqs'][f"peak_{number}"],spec_dict['cl'],extent=(sim_params['startd'],sim_params['endd'],sim_params['starti'],sim_params['endi']),linewidths=5,colors=[c_colors[c_index],c_colors[c_index],c_colors[c_index]])
                    axins[1].contourf(Plot_dict[f"F_{alpha}"]['Two_freqs'][f"peak_{number}"],spec_dict['cl'],extent=(sim_params['startd'],sim_params['endd'],sim_params['starti'],sim_params['endi']),linewidths=5,colors=[c_colors[c_index],c_colors[c_index],c_colors[c_index]])
                    c_index += 1
            
            ax2.text(0.05, 0.95,f"$\\it{{\\alpha}}$ = {alpha}",horizontalalignment='left', verticalalignment='top',transform=ax2.transAxes,fontsize=48,fontweight='bold')
            ax2.set_xlabel('$^{1}$H ppm',fontsize=48,fontweight='bold')
            ax2.set_ylabel('$^{15}$N ppm',fontsize=48,fontweight='bold')
            ax2.set_yticks([123.1,123.0,122.9,122.8])
            ax2.set_xlim(sim_params['endd'],sim_params['startd'])
            ax2.set_ylim(sim_params['endi'],sim_params['starti'])
            ax5.set_xlabel('$L_{T}$/$P_{T}$',fontsize=48,fontweight='bold')
            ax5.set_ylabel('CSP [Hz]',fontsize=48,fontweight='bold')
            ax5.text(0.6, 0.85,f"$\\it{{\\alpha}}$ = {alpha}",horizontalalignment='left', verticalalignment='top',transform=ax5.transAxes,fontsize=48,fontweight='bold',color=fastcolors[index])
            axins[1].set_xlabel('$^{1}$H ppm')
            axins[1].set_ylabel('$^{15}$N ppm')
            axins[1].set_yticks([123.1,122.8])
            axins[1].set_xlim(sim_params['wUU_dir']+0.02,sim_params['wBB_dir']-0.02)
            axins[1].set_ylim(sim_params['wUU_ind']+0.05,sim_params['wBB_ind']-0.05)
                    
        if index == 2:
            for number,ligand in enumerate(sim_params['LT']):
                if number == 0 or number % 9 == 0:
                    ax6.plot(sim_params['LT']/sim_params['PT'],Plot_dict[f"F_{alpha}"]['Two_freqs']['CSP'],linewidth=5,color=fastcolors[index])
                    ax3.contourf(Plot_dict[f"F_{alpha}"]['Four_freqs'][f"peak_{number}"],spec_dict['cl'],extent=(sim_params['startd'],sim_params['endd'],sim_params['starti'],sim_params['endi']),linewidths=5,colors=[c_colors[c_index],c_colors[c_index],c_colors[c_index]])
                    axins[2].contourf(Plot_dict[f"F_{alpha}"]['Two_freqs'][f"peak_{number}"],spec_dict['cl'],extent=(sim_params['startd'],sim_params['endd'],sim_params['starti'],sim_params['endi']),linewidths=5,colors=[c_colors[c_index],c_colors[c_index],c_colors[c_index]])
                    c_index += 1
            
            ax3.text(0.05, 0.95,f"$\\it{{\\alpha}}$ = {alpha}",horizontalalignment='left', verticalalignment='top',transform=ax3.transAxes,fontsize=48,fontweight='bold')
            ax3.set_xlabel('$^{1}$H ppm',fontsize=48,fontweight='bold')
            ax3.set_ylabel('$^{15}$N ppm',fontsize=48,fontweight='bold')
            ax3.set_yticks([123.1,123.0,122.9,122.8])
            ax3.set_xlim(sim_params['endd'],sim_params['startd'])
            ax3.set_ylim(sim_params['endi'],sim_params['starti'])
            ax6.set_xlabel('$L_{T}$/$P_{T}$',fontsize=48,fontweight='bold')
            ax6.set_ylabel('CSP [Hz]',fontsize=48,fontweight='bold')
            ax6.text(0.6, 0.85,f"$\\it{{\\alpha}}$ = {alpha}",horizontalalignment='left', verticalalignment='top',transform=ax6.transAxes,fontsize=48,fontweight='bold',color=fastcolors[index])
            axins[2].set_xlabel('$^{1}$H ppm')
            axins[2].set_ylabel('$^{15}$N ppm')
            axins[2].set_yticks([123.1,122.8])
            axins[2].set_xlim(sim_params['wUU_dir']+0.02,sim_params['wBB_dir']-0.02)
            axins[2].set_ylim(sim_params['wUU_ind']+0.05,sim_params['wBB_ind']-0.05)

        # Dimer slow exchange titration
        ax7.plot(sim_params['LT']/sim_params['PT'],Plot_dict[f"S_{alpha}"]['U_app'],'--',linewidth=5,color=slowcolors[index])
        ax7.plot(sim_params['LT']/sim_params['PT'],Plot_dict[f"S_{alpha}"]['B_app'],linewidth=5,color=slowcolors[index])

        ax7.set_ylabel('Intensity',fontsize=48,fontweight='bold')
        if index == 0:
            ax7.text(0.6, 0.69,f"$\\it{{\\alpha}}$ = {alpha}",horizontalalignment='left', verticalalignment='top',transform=ax7.transAxes,fontsize=44,fontweight='bold',color=slowcolors[index])
        if index == 1:
            ax7.text(0.6, 0.54,f"$\\it{{\\alpha}}$ = {alpha}",horizontalalignment='left', verticalalignment='top',transform=ax7.transAxes,fontsize=44,fontweight='bold',color=slowcolors[index])
        if index == 2:
            ax7.text(0.6, 0.41,f"$\\it{{\\alpha}}$ = {alpha}",horizontalalignment='left', verticalalignment='top',transform=ax7.transAxes,fontsize=44,fontweight='bold',color=slowcolors[index])
        ax7.set_xlabel('$L_{T}$/$P_{T}$',fontsize=48,fontweight='bold')
        ax7.set_xticks([0,10,20,30,40,50])
    
    fig1.tight_layout()
    fig1.savefig('Fig2_panelA_biggerlabels_italics.png',format='png')
    fig2.tight_layout()
    fig2.savefig('Fig2_panelB_biggerlabels_italics.png',format='png')
    fig3.tight_layout()
    fig3.savefig('Fig2_panelC_biggerlabels_italics.png',format='png')
    fig4.tight_layout()
    fig4.savefig('Fig2_panelD_biggerlabels_italics.png',format='png')
    fig5.tight_layout()
    fig5.savefig('Fig2_panelE_biggerlabels_italics.png',format='png')
    fig6.tight_layout()
    fig6.savefig('Fig2_panelF_biggerlabels_italics.png',format='png')
    fig7.tight_layout()
    fig7.savefig('Fig2_panelG_biggerlabels_italics.png',format='png')
    plt.close()

main()