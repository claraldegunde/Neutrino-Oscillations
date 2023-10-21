#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:07:03 2022

@author: claraaldegundemanteca
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec
font = {'family' : 'serif',
        'weight' : 'regular',
        'size'   : 14}

plt.rc('font', **font)


# DATA

"""
-Energies 0-10 GeV
- 200 bins
- 0.05 GeV step
- data.txt gives number of entries at each 0.05 GeV step
"""

# Obseved, experimental data 
data = np.loadtxt('data.txt', delimiter='\t', skiprows=0) #200 entries, one per bin

# Simmulation, if there was no neutrino oscillations 
unoscillated = np.loadtxt('unoscillatingflux.txt') #200 entries, one per bin

# Set energy scale 
n_bins = 200 #200 for a binwidth of 0.05
E=np.arange(0,10.05, 0.05)[1:]
binwidth = (E[-1]-E[0])/n_bins

def plot_data_and_simulation (recorded, simulation):
    """
    Given data files
    Plots the recorded data and the simulated unoscillated neutrinos as histograms
    """
    plt.figure()
    plt.bar(E, simulation, align='edge', width=binwidth, color='#002395',alpha=0.9, label='Unoscillated simulation') 
    plt.bar(E, recorded, align='edge', width=binwidth, color='#F9A602',alpha=1, label='Recorded data', linewidth=0.5) 
    plt.xlabel('Energy (GeV)',fontsize=12)
    plt.ylabel('Number of entries', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.7, linewidth=0.3)
    plt.savefig('Data and sim.png')
       
print(plot_data_and_simulation(data,unoscillated))


#PDF DISTRIBUTION FOR UNOSCILLATED

def P_mu (E, L, theta23, deltam2):
    """
    Returns the probability of a neutrino not oscillating, given E, L and oscillation parameters
    P of muon neutrino staying as a muon neutrino
    """

    return 1-((np.sin(2*theta23))**2)*(np.sin((1.267*deltam2*L)/E))**2

# Plot unoscillated PDF, for all energies from 0 to 10 GeV 
plt.figure()
plt.plot(E,P_mu (E,theta23=np.pi/4,deltam2=2.4*10**(-3),L=295.0), color='#002395', label=('${\\theta}_{23}$ =$\pi$/4, ${\\Delta}m^2_{23}$ = $2.4 · 10^{-3}$ '))
plt.xlabel('E (GeV)',fontsize=12)
plt.ylabel('Probability',fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.savefig('PDF unosc.png')



#CHANGING OSCILLATION PARAMETERS 

gs = gridspec.GridSpec(1,2)

"""
CHANGE THETA23
- Theta 23 goes from 0 to pi/2
- In both cases PDF=1
- Increasing theta23 increases amplitude

"""
ax = plt.subplot(gs[0, 0])
plt.plot(E,P_mu (E,theta23=0,deltam2=2.4*10**(-3),L=295.0), color='#002395', label=('${\\theta}_{23}$=0'))
# plt.plot(E,P_mu (E,theta23=np.pi/8,deltam2=2.4*10**(-3),L=295.0), label=('${\\theta}_{23}$ =$\pi$/8'))
plt.plot(E,P_mu (E,theta23=np.pi/6,deltam2=2.4*10**(-3),L=295.0), color='#F9A602',label=('${\\theta}_{23}$ =$\pi$/6'))
plt.plot(E,P_mu (E,theta23=np.pi/4,deltam2=2.4*10**(-3),L=295.0), color='red', label=('${\\theta}_{23}$ =$\pi$/4'))
plt.plot(E,P_mu (E,theta23=np.pi/2,deltam2=2.4*10**(-3),L=295.0), color='green',label=('${\\theta}_{23}$ = $\pi$/2'))
plt.xlabel('E (GeV)',fontsize=12)
plt.ylabel('Probability',fontsize=12)
plt.legend(fontsize=10)
plt.grid()

"""
CHANGE DELTAM2
- Less frequency if we increase deltam^2
- Function widens
"""

ax = plt.subplot(gs[0, 1])
plt.plot(E,P_mu (E,theta23=0,deltam2=0,L=295.0),color='#002395', label=(' ${\\Delta}m^2_{23}$=0'))
plt.plot(E,P_mu (E,theta23=np.pi/4,deltam2=3*10**(-3),L=295.0),color='#F9A602', label=(' ${\\Delta}m^2_{23}$ =$2.4·10{-3}$'))
# plt.plot(E,P_mu (E,theta23=np.pi/4,deltam2=4*10**(-3),L=295.0), label=(' ${\\Delta}m^2_{23}$ =$4·10{-3}$'))
plt.plot(E,P_mu (E,theta23=np.pi/4,deltam2=6*10**(-3),L=295.0), color='green', label=(' ${\\Delta}m^2_{23}$ =$6·10{-3}$'))
plt.xlabel('E (GeV)',fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.savefig('PDFs.png')



#LIKELIHOOD FUNCTION (NLL(theta23,deltam2))

"""
Lambda = PDF x simulation, the expected no. of entries,
"""

L=295.0

def NLL (theta23,deltam2): 
    """
    Returns NLL given a theta23 and deltam2
    """
    lambd = P_mu (E, L, theta23, deltam2) * unoscillated
    mi=data
    sum = 0
    for i in range (len(lambd)): #sum over bins starting from bin 1
        sum += lambd[i]-mi[i]*np.log(lambd[i])
    return sum

def NLL_alpha (theta23,deltam2, alpha=1): 
    """
    Now including alpha
    Returns NLL given a theta23 and deltam2 and alpha
    Lok at the equation, here alpha is the RATE OF CHANGE
    When alpha=1 it's the same so can use this in general 
    """
    lambd_new = P_mu (E, L, theta23, deltam2) * alpha *unoscillated  
    mi=data
    sum = 0
    for i in range (len(lambd_new)): #sum over bins starting from bin 1
        sum += lambd_new[i]-mi[i]*np.log(lambd_new[i])
    return sum

"""
When alpha =1, same
"""
deltam2=2.4e-3 #Use predictions above for initial guesses
theta23_range=np.linspace(0,np.pi/2,200)
NLL_array=[]
for i in theta23_range:
    NLL_array.append(NLL(i,deltam2))
    
    
    
# INVESTIGATING CHANGES ON NLL CHANGING ANGULAR PARAMETERS INDEPENDENTLY

#Fix deltam2 and change theta23. Record NLL 
deltam2=2.4e-3 #Use predictions above for initial guesses
theta23_range=np.linspace(0,np.pi/2,n_bins)
NLL_array_theta23=[]
for i in theta23_range:
    NLL_array_theta23.append(NLL(i,deltam2))
    
plt.figure()
plt.plot(theta23_range,NLL_array_theta23)
plt.title('${\\Delta}m^2_{23}$ = $%f$' % (deltam2))
plt.ylabel('NLL')
plt.xlabel('${\\theta}_{23}$')
plt.grid()

#Fix theta23 and change deltam2. Record NLL
theta23=np.pi/4
deltam2_range=np.linspace(0,10e-3,200)
NLL_array_deltam2=[]
for i in deltam2_range:
    NLL_array_deltam2.append(NLL(theta23,i))
    
plt.figure()
plt.plot(deltam2_range,NLL_array_deltam2)
plt.title('${\\theta}_{23}$= $%f$' % (theta23))
plt.ylabel('NLL')
plt.xlabel('${\\Delta}m^2_{23}$')
plt.grid()

#Fix theta23 and deltam2 and change alpha. Record NLL
alpha_range=np.linspace(0,3, n_bins) 
NLL_array_alpha=[]
for i in alpha_range:
    NLL_array_alpha.append(NLL_alpha(theta23,deltam2,i))
    
plt.figure()
plt.plot(alpha_range,NLL_array_alpha)
plt.title('${\\Delta}m^2_{23}$ = $%f$' % (deltam2))
plt.ylabel('NLL')
plt.xlabel('${\\theta}_{23}$')
plt.grid()


#Fix theta23 and deltam2 and change alpha. Record NLL 
theta23=np.pi/4
alpha=1.77
deltam2_range=np.linspace(0,10e-3,200)
NLL_array_deltam2=[]
for i in deltam2_range:
    NLL_array_deltam2.append(NLL_alpha(theta23,i, alpha))
    
plt.figure()
plt.plot(deltam2_range,NLL_array_deltam2)
plt.title('${\\theta}_{23}$= $%f$' % (theta23))
plt.ylabel('NLL')
plt.xlabel('${\\Delta}m^2_{23}$')
plt.grid()



# 3D VISUALISATION

theta23_range=np.linspace(0, np.pi/2, 200)
deltam2_range=np.linspace(0,2e-2, 200)

X, Y= np.meshgrid(theta23_range, deltam2_range)

Z=np.zeros((200,200))

for i in range(0, len(theta23_range)):
    for j in range(0,len(deltam2_range)):
        Z[j][i]=NLL(theta23_range[i],deltam2_range[j])

# fig=plt.figure()
# ax=plt.axes(projection='3d')
# ax.plot_surface(X, Y,Z, cmap='cividis')
# ax.set_xlabel('${\\theta}_{23}$', labelpad=20, fontsize=12)
# ax.set_ylabel('${\\Delta}m^2_{23}$', labelpad=20, fontsize=12)
# ax.set_zlabel('NLL', labelpad=20, fontsize=12)
# plt.show()

plt.figure()
plt.contourf(X, Y,Z, cmap='jet_r')
plt.colorbar()
plt.xlabel('${\\theta}_{23}$', labelpad=20,fontsize=12)
plt.ylabel('${\\Delta}m^2_{23}$', labelpad=20, fontsize=12)
plt.savefig('Contour.png')


