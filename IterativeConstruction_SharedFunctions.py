#!/usr/bin/env python
# coding: utf-8

# # Shared functions
# 
# This notebook contains code for an iterative construction of optimal balanced growth states in metabolic networks. Codeblocks below detail how optimal balanced growth states can be found in practice. For additional illustrations, convert the marked markdown-blocks to code. All blocks should be executed in the shown order.

# ### General remarks:
# 
# Variables are named as in the paper. However, the indexing of metabolites, fluxes and enzymes is reversed. Due to the iterative construction adding metabolite producing enzymes at the start of the pathway, indexing in here starts at 0 at the final reaction, and increases upstream up until \ell-1.
# 
# For this reason, the environmental nutrient concentration is often referred to as nu - so that it's name is independent of the lenght of the pathway considered.
# 
# If variables are not defined as complex numbers, warnings which state that roots of negative numbers were computed may appear. They do not affect the performance of the code.

# In[1]:


### import of required packages:

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#import scipy as scp
#from scipy.optimize import curve_fit

#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import seaborn as sns
#import pandas as pd


# First, we define simple functions as provided in the main text of the manuscript, which relate state variables to each other

# In[2]:


def calculateRho(jin, jout, lam):
    """Returns the steady state metabolite concentration for a given production flux jin and consumption 
    flux jout at a growth rate lam"""
    return (jin-jout)/lam

def calculatePhi(jout, rho, kappa, KM):
    """Returns the enzyme mass-fraction phi required to catalyze a production flux jout at a metabolite
    concentration rho"""
    return jout/kappa*(1+KM/rho)

def calculatejout(rho, phi, kappa, KM):
    """Michaelis-Menten-Enzymes with a mass fraction phi use substrates present at a concentration rho
    with a substrate affinity KM at a maximal rate kappa to catalyze a mass flux which is returned"""
    return phi*kappa/(1+KM/rho)

def calculatebetas(phis, jouts):
    """Returns metabolite production costs for a given set of enzyme mass fractions phis and fluxes jouts"""
    betas = np.cumsum(phis[::-1]*jouts[0]/jouts[::-1])
    betas = np.append(betas[::-1], 0)
    return betas[1:]

def calculateFluxes(rhos, lam):
    """Returns metabolite fluxes for a given vector of metabolite concentrations and growth rate lam."""
    return lam*(np.cumsum(rhos)-rhos+1)

def findPhis(kappas, KMs, rhos):
    """Calculates a phi-vector for given rhos and kinetic parameters. If no matching phis exist, returns partly nans"""
    jlocals2 = [1]
    for k in range(np.shape(kappas)[0]-1):
        if rhos[k]>0:
            jlocals2 = np.append(jlocals2, jlocals2[-1]+jlocals2[0]*rhos[k])
        else:
            jlocals2 = np.append(jlocals2, np.nan)
            
    phis = jlocals2/kappas*(1+KMs/rhos)
    jlocals2 = [1/np.nansum(phis)]
    for k in range(np.shape(kappas)[0]-1):
        if rhos[k]>0:
            jlocals2 = np.append(jlocals2, jlocals2[-1]+jlocals2[0]*rhos[k])
        else:
            jlocals2 = np.append(jlocals2, np.nan)
            
    phis = jlocals2/kappas*(1+KMs/rhos)
    #print(np.sum(phis))
    return phis


# Next, we define a simple simulation-function, which calculates the growth rate of a given metabolic state specified by the environmental nutrient concentration and all enzyme expression levels. This function can be used to cross-check iteratively computed optimal balanced growth states.

# In[3]:


###Forward-simulation functions returning lambda for a given set of phis for crosschecking optimality

def simulate(phis, kappas, KMs, nu=1, lamapprox=1):
    """Iterates through a chain with given phis and nu to calculate lambda"""
    #this simulation must not take lam as a fixed parameter - here it only takes a starting value for a
    #numerical iteration as forward-calculation of the chain is only possible if lambda is known
    
    #This function requires a normalized set of enzyme mass fractions
    phis = phis/np.sum(phis)
    
    
    jinInitial = kappas[-1]*phis[-1]/(1+KMs[-1]/nu) #compute the uptake flux of nutrients
    lamLocal = lamapprox
    Nenzymes = np.shape(kappas)[0]
    for j in range(Nenzymes): #numerically iterate a chain of lenght \ell \ell times.
        jin = jinInitial
        
        for i in np.arange(np.shape(phis)[0])[:Nenzymes-1][::-1]: #iterate through the chain, starting with the next enzyme after nutrient uptake.
            #initialize local variables
            phi = phis[i]
            kappa = kappas[i]
            KM = KMs[i]

            #compute the steady state concentration of the next metabolite
            rho = (jin-KM*lamLocal-kappa*phi+np.sqrt(4*jin*KM*lamLocal+(jin-KM*lamLocal-kappa*phi)**2))/(2*lamLocal)
            
            #compute the flux through the next reaction
            jin = jin-lamLocal*rho

        #the growth rate is then simply given by the flux through the last reaction, as sum(phis) = 1
        #To ensure convergence of the computed growth rate, growth rate is not updated with the value calculated in the last loop
        #iteration, but instead just moved closer to the correct value.
        lamLocal = (jin+lamLocal)/2
    return lamLocal

def simulateInDetail(phis, args):
    """Additionally returns phis, rhos and js from given phis, otherwise behaves like the simulate function"""
    #args should have a shape of np.ones(2*Nenzymes+2)
    Nenzymes = int((np.shape(args)[0]-2)/2)
    kappas = args[0:Nenzymes]
    KMs = args[Nenzymes:2*Nenzymes]
    nu = args[2*Nenzymes]
    lamapprox = simulate(phis, kappas, KMs, nu)
    
    jin = kappas[-1]*phis[-1]/(1+KMs[-1]/nu)
    lamLocal = lamapprox
    
    rhos = [nu]
    jouts = [jin]
    for i in np.arange(np.shape(phis)[0])[:Nenzymes-1][::-1]:
        phi = phis[i]
        kappa = kappas[i]
        KM = KMs[i]

        rho = (jin-KM*lamLocal-kappa*phi+np.sqrt(4*jin*KM*lamLocal+(jin-KM*lamLocal-kappa*phi)**2))/(2*lamLocal)
        jin = jin-lamLocal*rho
        
        rhos.append(rho)
        jouts.append(jin)
    if np.abs(jin-lamapprox)/lamapprox>0.05:
        print("Numerical computation of growth rate did not converge sufficiently. Try again with a better estimate of growth rate.")
    return phis, np.array(rhos)[::-1], np.array(jouts)[::-1]

def wrapper(logphis, args):
    """wrapper separating variables and parameters of the simulate function, and using ln(phi) as inputs.
    Returns negative lambda"""
    Nenzymes = int((np.shape(args)[0]-2)/2)
    phis = np.exp(logphis)
    kappas = args[0:Nenzymes]
    KMs = args[Nenzymes:2*Nenzymes]
    nu = args[2*Nenzymes]
    lamapprox = args[2*Nenzymes+1]
    lam = simulate(phis, kappas, KMs, nu, lamapprox)
    return -np.real(lam/np.sum(phis))


# Below is the code for finding optimal balanced growth states by iteration. In contrast to the manuscript, steady state metabolite concentrations are used as independent variables for ease of notation. They are directly linked to fluxes via the calculateFluxes function.

# In[4]:


#Internal iteration before an environmental boundary condition is added
def rhoIteratorInternal(kappas, KMs, rho0approx):
    """Internal iterator, which calculates optimal metabolite concentrations rho, based on a set of kinetic
    parameters and a starting value rho_ell"""
    rhoq = rho0approx
    rhos = np.array([rhoq])
    NEnzymes = np.shape(kappas)[0]
    #print(NEnzymes)
    for q in range(NEnzymes-2):
        # \rho_{q+1}^2\bigg(\frac{K_M^q}{\rho_q^2}\frac{1+\sum_j^{q-1} \rho_j}{\kappa_q}-\frac{1}{\kappa_{q+1}}\bigg)
        #-\rho_{q+1}  \bigg(\frac{K_M^{q+1}}{\kappa_{q+1}}\bigg)
        #-\bigg(\frac{K_M^{q+1}}{\kappa_{q+1}}(1+\sum_j^{q} \rho_j)\bigg)\\

        sumqminus = np.sum(rhos[:q])
        sumq = np.sum(rhos[:q+1])
        
        a = KMs[q]/rhos[q]**2*(1+sumqminus)/kappas[q]-1/kappas[q+1]
        b = -KMs[q+1]/kappas[q+1]
        c = -KMs[q+1]/kappas[q+1]*(1+sumq)
        
        rhoq = (-b+np.sqrt(b**2-4*a*c))/2/a
        rhos = np.append(rhos, rhoq)       
    return rhos

#Construction of a boundary condition under which the iteration start is growth optimal
def BCIterator(kappas, KMs, rho0):
    """Calculates the optimal metabolic state of a cell from all kinetic parameters and a iteration 
    starting condition rho0approx"""
    rhos = rhoIteratorInternal(kappas, KMs, rho0)
    nu = KMs[-1]/(kappas[-1]/kappas[-2]*(1+np.sum(rhos[:-1]))*KMs[-2]/rhos[-1]**2-1)
    rhos = np.append(rhos, nu)
    return rhos

#Wrapper for varying the iteration start until the calculated environment matches a given environment
def wrapper4(logrho0, args):
    """Returns a number, which is large when there is a discrepancy between the nutrient concentration specified,
    and the nutrient concentration constructed. Can be minimized to find the correct iteration starting condition."""
    Nenzymes = int((np.shape(args)[0]-2)/2)
    kappas = args[0:Nenzymes]
    KMs = args[Nenzymes:2*Nenzymes]
    nu = args[2*Nenzymes]
    lamapprox = args[2*Nenzymes+1]
    
    rhos = BCIterator(kappas, KMs, np.exp(logrho0))
    phis = findPhis(kappas, KMs, rhos)
    
    if np.isnan(phis[-1]) or rhos[-1]<0:
        return np.sum(np.isnan(phis))+1-1/(1+np.nansum(rhos[rhos>0]))+np.any(rhos<0)
    else:
        return (nu/(2*rhos[-1]))**2-nu/(2*rhos[-1])
        #return -nu/rhos[-1]*np.log(rhos[-1]*np.exp(1)/nu)
        #return -rhos[-1]*(1/KMmean+1/nu)*np.exp(-rhos[-1]/nu)


# #### Convert this markdown to code to verify construct an optimal balanced growth state for an exemplary set of kinetic parameters
# 
# #Creating and Cross-checking the solution for an exemplary parametrization:
# Nenzymes = 5
# 
# kappamean = 130
# KMmean = .0001
# 
# kappas = [2.28611997e+02, 1.42736287e+02, 6.69505415e+01, 1.35961125e+02, 5.05010591e+01] #np.random.lognormal(np.log(kappamean), 1, Nenzymes) 
# KMs = [4.24844752e-05, 3.54447218e-02, 1.14051378e-03, 1.20899641e-03, 9.99516462e-05]#np.random.lognormal(np.log(KMmean), 2.5, Nenzymes)
# 
# optrhos = BCIterator(kappas, KMs, np.exp(-6.28640785))
# print("Nutrient condition in which the state is optimal (given as saturation of the first enzyme):", optrhos[-1]/KMs[-1])

# #### Convert this markdown to code to verify that the iterative solution found above is indeed an optimal balanced growth state.
# 
# #We first calculate the corresponding proteome resource allocation, which matches the growth optimal metabolite concentrations
# optphis = findPhis(kappas, KMs, optrhos)
# #print("This value must be one:", np.sum(optphis)) #Sanity check of the solution
# 
# #then we determine the corresponding growth rate
# lamapprox = simulate(optphis, kappas, KMs, optrhos[-1], lamapprox=16)
# #print("The growth rate of the exemplary solution is:", lamapprox, "/h")
# 
# #Lastly, we test whether altered resource allocations have a lowered growth rate.
# relativeExpressions = np.linspace(0.95, 1.05, 100)
# 
# for i in range(5):
#     output = []
#     for relativeExpression in relativeExpressions:
#         testphis = np.copy(optphis)
#         testphis[i] = testphis[i]*relativeExpression
#         testphis = testphis/np.sum(testphis)
#         output.append(simulate(testphis, kappas, KMs, optrhos[-1], lamapprox=lamapprox)/lamapprox)
#     plt.plot(relativeExpressions, output, label = "Enzyme %d"%i)
#     
# plt.ylim(0.99, 1.002)
# plt.xlim(0.95, 1.05)
# plt.ylabel("Relative growth rate")
# plt.xlabel("Relative proteine expression level")
# plt.legend(title = "Varied expression level")
# #print(optphis)

# #An algorithm based on the functions defined above is inefficient in finding an iteration start which matches a given environment. 
# 
# #### Convert this markdown to code, to see, that a given environmental nutrient concentration can not be matched efficiently numerically by an optimization based on constructed boundary conditions
# Illustration of the inefficient minimization:
# res = minimize(wrapper4, -5, args = args) ### does not converge
# print(res, args)

# Instead we construct all but the environmental nutrient concentration iteratively, add the given environment and then pursue a maximization of growth rate by variation of the iteration start

# In[5]:


#Adding a given boundary condition, the rho0approx needs to be varied numerically until the found state is optimal
def rhoIterator(kappas, KMs, nu, rho0approx):
    rhos = rhoIteratorInternal(kappas, KMs, rho0approx)
    rhos = np.append(rhos, nu)
    return rhos

#Wrapper of rhoIterator, which can be used to numerically find the iteration start in which the given nutrient environment is optimal
def wrapper3(logrho0, args):
    """wrapper separating variables and parameters of the simulate function, and using ln(rho_0) as input.
    Returns a measure of negative lambda, which is corrected for the average enzyme efficiency to ensure
    robust optimization"""
    #print(logrho0)
    Nenzymes = int((np.shape(args)[0]-2)/2)
    kappas = args[0:Nenzymes]
    KMs = args[Nenzymes:2*Nenzymes]
    nu = args[2*Nenzymes]
    lamapprox = args[2*Nenzymes+1]
    
    rhos = rhoIterator(kappas, KMs, nu, np.exp(logrho0))
    phis = findPhis(kappas, KMs, rhos)
    
    if np.isnan(phis[-1]):
        return np.sum(np.isnan(phis))+1-1/(1+np.nansum(rhos[rhos>0]))+np.any(rhos<0)
    else:
        lam1 = kappas[-1]*phis[-1]/(1+KMs[-1]/rhos[-1])*1/(1+np.sum(rhos[:-1]))
        #lam2 = kappas[0]*phis[0]/(1+KMs[0]/rhos[0])
        #lam = simulate(phis, kappas, KMs, nu, lam1)
        #print(lam1, lam2, lam)
        return -lam1/np.sum(phis)*np.sum(1/kappas)


# #### Convert this markdown to code, to see that an optimization based on this scheme converges to the correct iteration starting value:
# 
# args = np.array([*kappas, *KMs, optrhos[-1], 1])
# 
# res = minimize(wrapper3, -5, args = args)
# print("Numerically determined iteration start:", np.exp(res.x[0]))
# print("True iteration start:                  ", optrhos[0])

# In[6]:


### definition of additional simple functions used throughout the notebook

def linear(xs, a, b):
    """A simple linear function used for fitting"""
    return a*xs + b

