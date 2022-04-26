### Simulate and/or fit homo-oligomerization models for DLS ###

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

# Model for generating simulated DLS data for monomer to trimer equilibrium
def MonomerTrimer(CT,Mass,D_Monomer,D_Trimer,K,k):
    
    Monomer = []
    Trimer = []
    
    for x in range(len(CT)):
        
        r = np.roots([3*K,0,1,-CT[x]])
        
        for y in range(len(r)):
            
            if np.isreal(r[y]) & np.greater(r[y],0):
                
                Monomer.append(r[y])
                Trimer.append((CT[x] - Monomer[x])/3)
    
    Dz = [(Mass**2*Monomer[x]*D_Monomer*(1 + k*Monomer[x]) + (3*Mass)**2*Trimer[x]*D_Trimer*(1 + k*Trimer[x]))/(Mass**2*Monomer[x] + (3*Mass)**2*Trimer[x]) for x in range(len(Monomer))]          

    return Dz

# Calculate monomer and trimer populations from optimized fitted parameters
def MonomerTrimerPopulations(CT,K):
    
    Monomer = []
    Trimer = []
    
    for x in range(len(CT)):
        
        r = np.roots([3*K,0,1,-CT[x]])
        
        for y in range(len(r)):
            
            if np.isreal(r[y]) & np.greater(r[y],0):
                
                Monomer.append(r[y])
                Trimer.append((CT[x] - Monomer[x])/3)
    
    P_Monomer = [Monomer[x]/CT[x] for x in range(len(CT))]
    P_Trimer = [3*Trimer[x]/CT[x] for x in range(len(CT))]

    return P_Monomer, P_Trimer

# Model for generating simulated DLS data for trimer to hexamer equilibrium
def TrimerHexamer(CT,Mass,D_Trimer,D_Hexamer,K,k):
    
    Trimer = [(-3 + np.sqrt(9 + 24*K*CT[x]))/(12*K) for x in range(len(CT))]
    Hexamer = [(CT[x] - 3*Trimer[x])/6 for x in range(len(CT))]
    
    Dz = [((3*Mass)**2*Trimer[x]*D_Trimer*(1 + k*Trimer[x]) + (6*Mass)**2*Hexamer[x]*D_Hexamer*(1 + k*Hexamer[x]))/((3*Mass)**2*Trimer[x] + (6*Mass)**2*Hexamer[x]) for x in range(len(CT))]
    
    return Dz

# Calculate trimer and hexamer populations from fitted parameters
def TrimerHexamerPopulations(CT,K):
    
    Trimer = [(-3 + np.sqrt(9 + 24*K*CT[x]))/(12*K) for x in range(len(CT))]
    Hexamer = [(CT[x] - 3*Trimer[x])/6 for x in range(len(CT))]
                
    P_Trimer = [3*Trimer[x]/CT[x] for x in range(len(CT))]
    P_Hexamer = [6*Hexamer[x]/CT[x] for x in range(len(CT))]
        
    return P_Trimer, P_Hexamer

################################################################################################################
####### Set up concentrations, molecular weight of the monomer, model to be simulated/fit ######################
### Set up diffusion coefficients at infinite dilution, non-ideality parameter, and noise for simulated data ###
CT_sim = np.logspace(-5,-3,15)
#CT_sim = np.array([10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000])*1e-6 # for experimental data
Mass = 1
K = 5e8
D_Monomer = 6e-7
D_Trimer = 5.5e-7
D_Hexamer = 4.5e-7
k = -200
modeling = 'MonomerTrimer'
noise = np.random.normal(1e-8,1e-9,len(CT_sim))
#noise = np.zeros(len(CT_sim))
################################################################################################################
################################################################################################################
################################################################################################################

# Run simulations for monomer-trimer equilibrium, add noise, then fit simulated data
if modeling == 'MonomerTrimer':
    
    # Set up model and parameters
    mod = Model(MonomerTrimer)
    mod_params = mod.make_params(Mass=Mass,D_Monomer=D_Monomer,D_Trimer=D_Trimer,K=K,k=k)
    mod_params['Mass'].vary = 'False'
    mod_params['k'].vary = 'False'
    
    # Simulate data according to input parameters
    Dz_sim = mod.eval(CT=CT_sim,Mass=Mass,D_Monomer=D_Monomer,D_Trimer=D_Trimer,K=K,k=k)
    #Dz_sim = [Dz_sim[x] + np.random.normal(0.0001*Dz_sim[x],0.00005*Dz_sim[x],1) for x in range(len(Dz_sim))]
    noise = np.
    
    Dz_sim = np.array(Dz_sim) + noise
    
    # Fit simulated data
    ### PUT EXPERIMENTAL DATA HERE FOR FITTING! ###
    #Dz_sim = np.array([])
    out = mod.fit(np.array(Dz_sim),mod_params,CT=CT_sim)
    
    # Get optimized parameters and fit errors
    opt_D_Monomer = out.params["D_Monomer"].value
    opt_D_Monomer_err = out.params['D_Monomer'].stderr
    opt_D_Trimer = out.params["D_Trimer"].value
    opt_D_Trimer_err = out.params["D_Trimer"].stderr
    opt_K = out.params['K'].value
    opt_K_err = out.params['K'].stderr
    
    # Get populations of monomer and trimer
    P_Monomer, P_Trimer = MonomerTrimerPopulations(CT_sim,opt_K)
    P_total = [P_Monomer[x] + P_Trimer[x] for x in range(len(P_Monomer))]
    
    # Plot simulated data and optimal fit, list optimized parameters in top right
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(CT_sim*1e6,Dz_sim,'ko')
    ax1.plot(CT_sim*1e6,out.best_fit,'r--')
    ax1.set_ylabel('$D_{avg}$ $cm^{2}$ $s^{-1}$')
    ax1.set_xlabel('[Monomer] $\mu$M')
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax1.text(0.3,0.95,"$D_{Monomer}$ = %.2e $\pm$ %.2e $cm^{2}$ $s^{-1}$\n$D_{Trimer}$ = %.2e $\pm$ %.2e $cm^{2}$ $s^{-1}$\nK = %.2e $\pm$ %.2e $M^{-2}$"%(opt_D_Monomer,opt_D_Monomer_err,opt_D_Trimer,opt_D_Trimer_err,opt_K,opt_K_err),transform=ax1.transAxes,va="top")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(CT_sim*1e6,P_Monomer,label='Monomer')
    ax2.plot(CT_sim*1e6,P_Trimer,label='Trimer')
    ax2.plot(CT_sim*1e6,P_total,label='Sum')
    ax2.set_ylabel('Population')
    ax2.set_xlabel('[Monomer] $\mu$M')
    ax2.set_ylim(-0.01,1.01)
    ax2.legend(frameon=False)

# Run simulations for trimer-hexamer equilibrium, add noise, then fit simulated data
elif modeling == 'TrimerHexamer':
    
    # Set up model and parameters
    mod = Model(TrimerHexamer)
    mod_params = mod.make_params(Mass=Mass,D_Trimer=D_Trimer,D_Hexamer=D_Hexamer,K=K,k=k)
    mod_params['Mass'].vary = 'False'
    mod_params['k'].vary = 'False'
    #mod_params['D_Trimer'].vary='False'
    #mod_params['D_Hexamer'].vary='False'
    
    # Simulate data according to input parameters
    Dz_sim = mod.eval(CT=CT_sim,Mass=Mass,D_Trimer=D_Trimer,D_Hexamer=D_Hexamer,K=K,k=k)
    Dz_sim = Dz_sim + noise
    
    # Fit simulated data
    out = mod.fit(Dz_sim,mod_params,CT=CT_sim)
    
    # Get optimized parameters and fit errors
    opt_D_Trimer = out.params["D_Trimer"].value
    opt_D_Trimer_err = out.params['D_Trimer'].stderr
    opt_D_Hexamer = out.params["D_Hexamer"].value
    opt_D_Hexamer_err = out.params["D_Hexamer"].stderr
    opt_K = out.params['K'].value
    opt_K_err = out.params['K'].stderr
    
    # Get populations of monomer and trimer
    P_Trimer, P_Hexamer = TrimerHexamerPopulations(CT_sim,opt_K)
    P_total = [P_Trimer[x] + P_Hexamer[x] for x in range(len(P_Trimer))]
    
    # Plot simulated data and optimal fit, list optimized parameters in top right
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(CT_sim*1e6,Dz_sim,'ko')
    ax1.plot(CT_sim*1e6,out.best_fit,'r--')
    ax1.set_ylabel('$D_{avg}$ $cm^{2}$ $s^{-1}$')
    ax1.set_xlabel('[Monomer] $\mu$M')
    ax1.yaxis.major.formatter._useMathText = True
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax1.text(0.3,0.95,"$D_{Trimer}$ = %.2e $\pm$ %.2e $cm^{2}$ $s^{-1}$\n$D_{Hexamer}$ = %.2e $\pm$ %.2e $cm^{2}$ $s^{-1}$\nK = %.2e $\pm$ %.2e $M^{-1}$"%(opt_D_Trimer,opt_D_Trimer_err,opt_D_Hexamer,opt_D_Hexamer_err,opt_K,opt_K_err),transform=ax1.transAxes,va="top")
    
    #ax1.savefig('MonomerTrimer_fit_results.pdf')
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(CT_sim*1e6,P_Trimer,label='Trimer')
    ax2.plot(CT_sim*1e6,P_Hexamer,label='Hexamer')
    ax2.plot(CT_sim*1e6,P_total,label='Sum')
    ax2.set_ylabel('Population')
    ax2.set_xlabel('[Monomer] $\mu$M')
    ax2.set_ylim(-0.01,1.01)
    ax2.legend(frameon=False)