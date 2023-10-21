#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 13:48:02 2022

@author: claraaldegundemanteca
"""
import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'serif',
        'weight' : 'regular',
        'size'   : 14}

plt.rc('font', **font)

#DATA

# Obseved, experimental data 
data = np.loadtxt('data.txt', delimiter = '\t', skiprows = 0) #200 entries, one per bin

# Simmulation, if there was no neutrino oscillations 
unoscillated = np.loadtxt('unoscillatingflux.txt') #200 entries, one per bin

# Set energy scale 
n_bins = 200 #200 for a binwidth of 0.05
E = np.arange(0,10.05, 0.05)[1:]
binwidth = (E[-1]-E[0])/n_bins

def plot_data_and_simulation (recorded, simulation):
  """
  Given data files
  Plots the recorded data and the simulated unoscillated neutrinos as histograms
  """
  plt.figure()
  plt.bar(E, simulation, align = 'edge', width = binwidth,color = '#002395',alpha = 0.9, label = 'Unoscillated simulation') 
  plt.bar(E, recorded, align = 'edge', width = binwidth, color = '#F9A602',alpha = 1, label = 'Recorded data', linewidth = 0.5) 
  plt.xlabel('Energy (GeV)',fontsize = 12)
  plt.ylabel('Number of entries', fontsize = 12)
  plt.legend(fontsize = 12)
  plt.grid(alpha = 0.7, linewidth = 0.3)
  plt.savefig('Data and sim.png')
  
 

#PDF DISTRIBUTION FOR UNOSCILLATED

def P_mu (E, L, theta23, deltam2):
  """
  Returns the probability of a neutrino not oscillating, given E, L and oscillation parameters
  P of muon neutrino staying as a muon neutrino
  """

  return 1-((np.sin(2*theta23))**2)*(np.sin((1.267*deltam2*L)/E))**2


#NLL FUNCTION

def NLL_alpha (theta23,deltam2, alpha = 1): 
  """
  Now including alpha
  Returns NLL given a theta23 and deltam2 and alpha
  Lok at the equation, here alpha is the RATE OF CHANGE
  When alpha = 1 it's the same so can use this in general 
  """
  lambd_new =  P_mu (E, L, theta23, deltam2) * alpha *unoscillated 
  mi = data
  sum  =  0
  for i in range (len(lambd_new)): #sum over bins starting from bin 1
      sum += lambd_new[i]-mi[i]*np.log(lambd_new[i])
  return sum

theta23_range = np.linspace(0,np.pi/2, 200)
deltam2_range = np.linspace(0, 2e-2, 200)
alpha_range = np.linspace(0, 3, 200)
L  =  295

def central_diff_1var(function,x_array):
  """
  Given a function and an array of values for the independent variable,
  returns the derivative at each point
  """
  step = 0.001 #
  derivative = np.zeros((len(x_array)))
  for i in range(0, len(x_array)):
      derivative[i] = (function(x_array[i]+step)-function(x_array[i]-step))/(2*step)
  return derivative

# def quadratic(x):
#  return x**2

# def sin (x):
#  return np.sin(x)

# def cos (x):
#  return np.cos(x)

# x = np.linspace(0,5,200)

# plt.figure()
# plt.plot(x,central_diff_1var(sin,x), linewidth = 5)
# plt.plot(x,cos(x), linewidth = 1, color = 'red')

def central_diff_theta23_whole(function,theta23_array, deltam2, alpha):
  """
  Differentiate wrt theta23, initial starting points deltam2 and alpha (initial theta23 is first of the array)
  """
  step = 0.001
  derivative = np.zeros((len(theta23_array)))
  for i in range(0, len(theta23_array)):
      derivative[i] = (function(theta23_array[i]+step, deltam2, alpha)-function(theta23_array[i]-step,deltam2, alpha))/(2*step)
  return derivative

def central_diff_theta23(function,theta23, deltam2, alpha):
  """
  Differentiate wrt theta23, initial starting points deltam2 and alpha (initial theta23 is first of the array)
  """
  step = 1e-3
  derivative = (function(theta23+step, deltam2, alpha)-function(theta23-step,deltam2, alpha))/(2*step)
  return derivative


def central_diff_deltam2(function,theta23, deltam2, alpha):
  """
  Differentiate wrt theta23, initial starting points deltam2 and alpha (initial theta23 is first of the array)
  """
  step = 1e-5
  derivative = (function(theta23,deltam2+step, alpha)-function(theta23,deltam2-step, alpha))/(2*step)
  return derivative


def central_diff_alpha(function,theta23, deltam2, alpha):
  """
  Differentiate wrt theta23, initial starting points deltam2 and alpha (initial theta23 is first of the array)
  """
  step = 1e-1
  derivative = (function(theta23,deltam2,alpha+step)-function(theta23, deltam2, alpha-step))/(2*step)
  return derivative

def gradient (theta23_initial, deltam2_initial, alpha_initial,no_displacements, lr):
  """
  Returns the parameters (theta23, deltam2, alpha) from the gradient descent 
  method, given an initial set of parameters from which minimisation starts.
  Also need to specify the number of displacements towards the minimum
  and learning rate lr (needs to be << 1)
  """
  theta23_array = np.zeros((no_displacements))
  deltam2_array = np.zeros((no_displacements))
  alpha_array = np.zeros((no_displacements))
  theta23_array[0] = theta23_initial
  deltam2_array[0] = deltam2_initial
  alpha_array[0] = alpha_initial #initial values
  for i in range (0,no_displacements-1): #no of displacements
    if central_diff_theta23(NLL_alpha, theta23_array[i], deltam2_array[i], alpha_array[i]) > 0:
        theta23_array[i+1] = theta23_array[i] - lr*central_diff_theta23(NLL_alpha, theta23_array[i], deltam2_array[i], alpha_array[i])
    else: 
     theta23_array[i+1] = theta23_array[i] + lr*central_diff_theta23(NLL_alpha, theta23_array[i], deltam2_array[i], alpha_array[i])
    if central_diff_deltam2(NLL_alpha, theta23_array[i+1], deltam2_array[i], alpha_array[i]) > 0:
        deltam2_array[i+1] = deltam2_array[i] - lr*central_diff_deltam2(NLL_alpha, theta23_array[i+1], deltam2_array[i], alpha_array[i])
    else: 
        deltam2_array[i+1] = theta23_array[i] + lr*central_diff_deltam2(NLL_alpha, theta23_array[i+1], deltam2_array[i], alpha_array[i])
    if central_diff_alpha(NLL_alpha, theta23_array[i+1], deltam2_array[i+1], alpha_array[i]) > 0:
        alpha_array[i+1] = alpha_array[i] - lr*central_diff_alpha(NLL_alpha, theta23_array[i+1], deltam2_array[i+1], alpha_array[i])
    else: 
        alpha_array[i+1] = alpha_array[i] + lr*central_diff_alpha(NLL_alpha, theta23_array[i+1], deltam2_array[i+1], alpha_array[i])
  return theta23_array[-1],deltam2_array[-1], alpha_array[-1]

theta23_initial = 0.8
deltam2_initial = 0.004
alpha_initial = 2

theta23_gradient = gradient (theta23_initial, deltam2_initial, alpha_initial,no_displacements = 1000, lr = 1e-5)[0]
deltam2_gradient = gradient (theta23_initial, deltam2_initial, alpha_initial,no_displacements = 1000, lr = 1e-5)[1]
alpha_gradient = gradient (theta23_initial, deltam2_initial, alpha_initial,no_displacements = 1000, lr = 1e-5)[2]



#PLOTS

#Initial situation
theta23 = theta23_initial
deltam2_range = np.linspace(0,10e-3,200)
NLL_array_deltam2 = []
for i in deltam2_range:
   NLL_array_deltam2.append(NLL_alpha(theta23,i))

plt.figure()
plt.subplot(2,1,1)
plt.plot(deltam2_range,NLL_array_deltam2,color = '#002395', label='At initial condition, ${\\theta}_{23}$ = 0.8')
plt.plot(0.0079,NLL_alpha(theta23,0.0079), 'o', markersize = 6, color = 'red', label = 'Gradient descent')
#Plot where it's getting stuck
# plt.plot(deltam2_gradient,NLL_alpha(theta23_initial,deltam2_gradient), 'o', markersize = 3, color = 'red', label = 'Result from gradient descent')
plt.ylabel('NLL', fontsize = 14)
# plt.xlabel('${\\Delta}m^2_{23}$',fontsize = 14)
plt.legend()
plt.grid()
# plt.savefig('Gradient1.png')

#Found minimum
theta23 = 0.05
NLL_array_deltam2 = []
for i in deltam2_range:
  NLL_array_deltam2.append(NLL_alpha(theta23,i))

#See deltam2 trend near the theta23 initial condition
plt.subplot(2,1,2)
plt.plot(deltam2_range,NLL_array_deltam2,color = '#002395', label=('${\\theta}_{23}$ = 0.05'))
#Plot where it's getting stuck
plt.plot(0.0079,NLL_alpha(theta23,0.0079), 'o', markersize = 6, color = 'red', label = 'Gradient descent')
plt.ylabel('NLL', fontsize = 14)
plt.xlabel('${\\Delta}m^2_{23}$',fontsize = 14)
plt.legend()
plt.grid()
plt.savefig('Gradient2.png')

"""
CAREFUL WITH INITIAL POINTS

Probabbly stuck on some local minima. Remember if theta23 is not small, deltam2 has a lot of local minima
"""



#PRINT RESULTS
print('From gradient descent method')
print('theta23 = ', theta23_gradient)
print('deltam2 = ', deltam2_gradient)
print('alpha = ', alpha_gradient)