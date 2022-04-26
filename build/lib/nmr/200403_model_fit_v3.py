# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:42:14 2019

@author: toyam
"""
import os 
import numpy as np
from scipy import optimize as opt

import matplotlib.pyplot as plt
from functools import partial
from lmfit import minimize, Parameters, fit_report
import pandas as pd

# Calculate monomer and trimer populations from optimized fitted parameters
def ffs_complex(q,p):
    
    # Unpack variables and constants
    # Use population to avoid numerical crush
    # PL = L/LT
    PH, PT, PTL, PTL2, PTL3, PHL, PHL2, PHL3, PHL4, PHL5, PHL6, PL = p # Variables
    CT, LT, K1, K2, K3, K4 = q # Constants
    
    f=LT/CT
    
    # Equations have to be set up as equal to 0
   
    eq1 = -1. + PT + PTL + PTL2 + PTL3 + PH + PHL+ PHL2+ PHL3+ PHL4+ PHL5 + PHL6  # Protein equation
    
   
    #eq2 = -LT + PL*LT + PTL*CT/3. + 2*PTL2*CT/3 + 3*PTL3*CT/3 + PHL*CT/6 + 2*PHL2*CT/6+ 3*PHL3*CT/6 + 4*PHL4*CT/6 + 5*PHL5*CT/6 + 6*PHL6*CT/6 # Ligand equation

    eq2 = -f + PL*f + PTL/3. + 2*PTL2/3. + PTL3 + PHL/6. + PHL2/3.+ PHL3/2. + 2*PHL4/3. + 5*PHL5/6. + PHL6 # Ligand equation

    eq3 = 2*K1*(PT**2)*CT - 3*PH # Trimer-Hexamer equilibrium
    eq4 = 3*K2*PT*PL*LT - PTL # Trimer binding first ligand
    eq5 = K2*PTL*PL*LT - PTL2 # Trimer binding second ligand
    eq6 = K2*PTL2*PL*LT - 3*PTL3 # Trimer binding third ligand
    eq7 = 2*K4*(PTL3**2)*CT - 3*PHL6 # Trimer binding third ligand

    eq8 = 6*K3*PH*PL*LT - PHL # hexamer binding first ligand
    eq9 = 5*K3*PHL*PL*LT -2*PHL2 # hexamer binding 2nd ligand
    eq10 = 4*K3*PHL2*PL*LT - 3*PHL3 # hexamer binding 3rd ligand
    eq11 = 3*K3*PHL3*PL*LT - 4*PHL4 # hexamer binding 4th ligand
    eq12 = 2*K3*PHL4*PL*LT - 5*PHL5 # hexamer binding 5th ligand
    eq13 = K3*PHL5*PL*LT - 6*PHL6 # hexamer binding 6th ligand
    #print (eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13)
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13]

#data import
data = pd.read_csv("titration_input.csv")
tinept = 0.0072


# Data modification
# read data
out_df = data

# Add the peak volume, in which the inpet loss is corrected. 
out_df['corr_volume']=0.

for i in out_df.index:  
    R=out_df.at[i,'R']
    out_df.at[i,'corr_volume']=out_df.at[i,'volume']*np.exp(tinept*R)

# Add the population.
out_df['population']=0.
    
for i in out_df.index:
    residue_temp=out_df.at[i,'assignment']
    conc_temp=out_df.at[i,'conc']
    
    totalvol=np.sum(out_df[(out_df['assignment']==residue_temp)
                 & (out_df['conc']==conc_temp)].corr_volume)
    
    out_df.at[i,'population']=out_df.at[i,'corr_volume']/totalvol


data=out_df

conc=np.array(data[(data['assignment']=="L377")&(data['oligomer']=="6mer")].conc)

L377_6mer=np.array(data[(data['assignment']=="L377")&(data['oligomer']=="6mer")].population)
L377_3mer=np.array(data[(data['assignment']=="L377")&(data['oligomer']=="3mer")].population)
L377_bound=np.array(data[(data['assignment']=="L377")&(data['oligomer']=="bound")].population)

M420_free=np.array(data[(data['assignment']=="M420")&(data['oligomer']=="free")].population)
M420_3merb=np.array(data[(data['assignment']=="M420")&(data['oligomer']=="bound3mer")].population)
M420_6merb=np.array(data[(data['assignment']=="M420")&(data['oligomer']=="bound6mer")].population)

I362_free=np.array(data[(data['assignment']=="I362")&(data['oligomer']=="free")].population)
I362_bound=np.array(data[(data['assignment']=="I362")&(data['oligomer']=="bound")].population)

## Only limited peptide conc range is used

endpoint=5
conc=conc[0:endpoint]
L377_6mer=L377_6mer[0:endpoint]
L377_3mer=L377_3mer[0:endpoint]
L377_bound=L377_bound[0:endpoint]
M420_free=M420_free[0:endpoint]
M420_3merb=M420_3merb[0:endpoint]
M420_6merb=M420_6merb[0:endpoint]

I362_free=I362_free[0:endpoint]
I362_bound=I362_bound[0:endpoint]


#define parameters minimize function
fit_params = Parameters()
fit_params.add('K1',value=51730.0385, vary=False, min=0)
fit_params.add('K2',value=0, vary=True, min=0)
fit_params.add('K3',value=0, vary=True, min=0)
fit_params.add('K4',value=3038.31553, vary=False, min=0)


CT=150E-6

def objective(fit_params):
    
    K1 = fit_params['K1'] 
    K2 = fit_params['K2'] 
    K3 = fit_params['K3'] 
    K4 = fit_params['K4'] 
    
    PT=np.empty_like(conc,dtype=float)
    PTL=np.empty_like(conc,dtype=float)
    PTL2=np.empty_like(conc,dtype=float)
    PTL3=np.empty_like(conc,dtype=float)

    PH=np.empty_like(conc,dtype=float)
    PHL=np.empty_like(conc,dtype=float)
    PHL2=np.empty_like(conc,dtype=float)
    PHL3=np.empty_like(conc,dtype=float)    
    PHL4=np.empty_like(conc,dtype=float)
    PHL5=np.empty_like(conc,dtype=float)
    PHL6=np.empty_like(conc,dtype=float)
    
    PL=np.empty_like(conc,dtype=float)
     
    for i in range(len(conc)):
        p=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        q = [CT,conc[i]*1E-6,K1,K2,K3,K4]
        ffs_partial = partial(ffs_complex,q)
        # Solutions are ordered according to how the initial guess vector is arranged
        PH[i], PT[i], PTL[i], PTL2[i], PTL3[i], PHL[i], PHL2[i], PHL3[i], PHL4[i], PHL5[i], PHL6[i], PL[i] =opt.root(ffs_partial,p,method='lm').x  
        
    residual=np.zeros(0)
    
    PHf=(6*PH + 5*PHL + 4*PHL2 + 3*PHL3 + 2*PHL4 + PHL5)/6.
    PTf=(3*PT + 2*PTL + PTL2)/3.
    PTb=(PTL + 2*PTL2 + 3*PTL3)/3.
    PHb=(PHL + 2*PHL2 + 3*PHL3 + 4*PHL4 + 5*PHL5 + 6*PHL6)/6.

    #L377
    residual=np.append(residual,L377_6mer-PHf)
    residual=np.append(residual,L377_3mer-PTf)
    residual=np.append(residual,L377_bound-(PTb+PHb))
    
    #M420
    residual=np.append(residual,M420_free-(PHf+PTf))
    residual=np.append(residual,M420_3merb-PTb)
    residual=np.append(residual,M420_6merb-PHb)
    
    #I362
    residual=np.append(residual,I362_free-(PHf+PTf))
    residual=np.append(residual,I362_bound-(PTb+PHb))  
    
    return residual
    


result = minimize(objective,fit_params,method="nelder")

print(fit_report(result))
with open("report.txt", 'w') as fh:
    fh.write(fit_report(result))
    
opt_params = result.params
opt_K1 = opt_params["K1"].value 
opt_K2 = opt_params["K2"].value 
opt_K3 = opt_params["K3"].value 
opt_K4 = opt_params["K4"].value 


# Add fit population
out_df['fit_population']=0.
    
for i in out_df.index:
    conc_temp=out_df.at[i,'conc']
    resi=out_df.at[i,'assignment']
    oligomer=out_df.at[i,'oligomer']
    p=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    q = [CT,conc_temp*1E-6,opt_K1,opt_K2,opt_K3,opt_K4]
    ffs_partial = partial(ffs_complex,q)
    # Solutions are ordered according to how the initial guess vector is arranged

    PH, PT, PTL, PTL2, PTL3, PHL, PHL2, PHL3, PHL4, PHL5, PHL6, PL  = opt.root(ffs_partial,p,method='lm').x
    
    PHf=(6*PH + 5*PHL + 4*PHL2 + 3*PHL3 + 2*PHL4 + PHL5)/6.
    PTf=(3*PT + 2*PTL + PTL2)/3.
    PTb=(PTL + 2*PTL2 + 3*PTL3)/3.
    PHb=(PHL + 2*PHL2 + 3*PHL3 + 4*PHL4 + 5*PHL5 + 6*PHL6)/6.
    
    if resi=='L377':
        if oligomer=='6mer':
            fit_pop=PHf
        elif oligomer=='3mer':
            fit_pop=PTf
        elif oligomer=='bound':
            fit_pop=PTb+PHb
            
    elif resi=='M420':
        if oligomer=='free':
            fit_pop=PHf+PTf
        elif oligomer=='bound3mer':
            fit_pop=PTb
        elif oligomer=='bound6mer':
            fit_pop=PHb            

    elif resi=='I362':
        if oligomer=='free':
            fit_pop=PHf+PTf
        elif oligomer=='bound':
            fit_pop=PTb+PHb

    out_df.at[i,'fit_population']=fit_pop
    
out_df.to_csv("out.csv")

# Plot the data
concsim=np.arange(0,1000,5)
 
PTsim=np.empty_like(concsim,dtype=float)
PTLsim=np.empty_like(concsim,dtype=float)
PTL2sim=np.empty_like(concsim,dtype=float)
PTL3sim=np.empty_like(concsim,dtype=float)

PHsim=np.empty_like(concsim,dtype=float)
PHLsim=np.empty_like(concsim,dtype=float)
PHL2sim=np.empty_like(concsim,dtype=float)
PHL3sim=np.empty_like(concsim,dtype=float)    
PHL4sim=np.empty_like(concsim,dtype=float)
PHL5sim=np.empty_like(concsim,dtype=float)
PHL6sim=np.empty_like(concsim,dtype=float)

PLsim=np.empty_like(concsim,dtype=float)


for i in range(len(concsim)):
    p=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    q = [CT,concsim[i]*1E-6,opt_K1,opt_K2,opt_K3,opt_K4]
    ffs_partial = partial(ffs_complex,q)
    # Solutions are ordered according to how the initial guess vector is arranged
    PHsim[i], PTsim[i], PTLsim[i], PTL2sim[i], PTL3sim[i], PHLsim[i], PHL2sim[i], PHL3sim[i], PHL4sim[i], PHL5sim[i], PHL6sim[i], PLsim[i]=opt.root(ffs_partial,p,method='lm').x
   

PHf_sim=(6*PHsim + 5*PHLsim + 4*PHL2sim + 3*PHL3sim + 2*PHL4sim + PHL5sim)/6.
PTf_sim=(3*PTsim + 2*PTLsim + PTL2sim)/3.
PTb_sim=(PTLsim + 2*PTL2sim + 3*PTL3sim)/3.
PHb_sim=(PHLsim + 2*PHL2sim + 3*PHL3sim + 4*PHL4sim + 5*PHL5sim + 6*PHL6sim)/6.


cmap=['orangered','orange','seagreen','skyblue','dodgerblue','black']


plt.plot(concsim,PHsim,label="H")
plt.plot(concsim,PHLsim,label="HL")
plt.plot(concsim,PHL2sim,label="HL2")
plt.plot(concsim,PHL3sim,label="HL3")
plt.plot(concsim,PHL4sim,label="HL4")
plt.plot(concsim,PHL5sim,label="HL5")
plt.plot(concsim,PHL6sim,label="HL6")
plt.legend()
plt.savefig("Hpopulation.pdf")
plt.clf()

plt.plot(concsim,PTsim,label="T")
plt.plot(concsim,PTLsim,label="TL")
plt.plot(concsim,PTL2sim,label="TL2")
plt.plot(concsim,PTL3sim,label="TL3")
plt.legend()
plt.savefig("Tpopulation.pdf")
plt.clf()

plotconc=np.max(conc)*1.2

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(concsim,PHf_sim,c=cmap[5])
ax1.plot(conc,L377_6mer,c=cmap[5],marker='o',ls="None",label="Free 6mer")
ax1.plot(concsim,PTf_sim,c=cmap[4])
ax1.plot(conc,L377_3mer,c=cmap[4],marker='o',ls="None",label="Free 3mer")
ax1.plot(concsim,PTb_sim+PHb_sim,c=cmap[0])
ax1.plot(conc,L377_bound,c=cmap[0],marker='o',ls="None",label="Bound 3mer+6mer")
ax1.set_ylabel('Population')
ax1.set_xlabel('Concentration [$\mu$M]')
ax1.set_xlim(0,plotconc)
ax1.yaxis.major.formatter._useMathText = True
ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#ax1.text(0.35,0.95,"$K_{eq}$ = %.2e $\pm$ %.2e [$M$]"%  
#(opt_K1,std_Keq),transform=ax1.transAxes,va="top")
ax1.legend()
plt.tight_layout()
plt.savefig("L377_fit.pdf")


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(concsim,PHf_sim+PTf_sim,c=cmap[2])
ax1.plot(conc,M420_free,c=cmap[2],marker='o',ls="None",label="Free 6mer+3mer")
ax1.plot(concsim,PTb_sim,c=cmap[1])
ax1.plot(conc,M420_3merb,c=cmap[1],marker='o',ls="None",label="Bound 3mer")
ax1.plot(concsim,PHb_sim,c=cmap[0])
ax1.plot(conc,M420_6merb,c=cmap[0],marker='o',ls="None",label="Bound 6mer")
ax1.set_ylabel('Population')
ax1.set_xlabel('Concentration [$\mu$M]')
ax1.set_xlim(0,plotconc)
ax1.yaxis.major.formatter._useMathText = True
ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#ax1.text(0.35,0.95,"$K_{eq}$ = %.2e $\pm$ %.2e [$M$]"%  
#(opt_K1,std_Keq),transform=ax1.transAxes,va="top")
ax1.legend()
plt.tight_layout()
plt.savefig("M420_fit.pdf")



fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(concsim,PHf_sim+PTf_sim,c=cmap[2])
ax1.plot(conc,I362_free,c=cmap[2],marker='o',ls="None",label="Free 6mer+3mer")
ax1.plot(concsim,PTb_sim+PHb_sim,c=cmap[0])
ax1.plot(conc,I362_bound,c=cmap[0],marker='o',ls="None",label="Bound 3mer+6mer")
ax1.set_ylabel('Population')
ax1.set_xlabel('Concentration [$\mu$M]')
ax1.set_xlim(0,plotconc)
ax1.yaxis.major.formatter._useMathText = True
ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#ax1.text(0.35,0.95,"$K_{eq}$ = %.2e $\pm$ %.2e [$M$]"%  
#(opt_K1,std_Keq),transform=ax1.transAxes,va="top")
ax1.legend()
plt.tight_layout()
plt.savefig("I362_fit.pdf")


plt.close('all')



    

    
 
