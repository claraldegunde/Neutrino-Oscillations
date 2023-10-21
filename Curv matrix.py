#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:12:54 2022

@author: claraaldegundemanteca
"""
import numpy as np

# Obseved, experimental data 
data = np.loadtxt('data.txt', delimiter='\t', skiprows=0) #200 entries, one per bin

# Simmulation, if there was no neutrino oscillations 
unoscillated = np.loadtxt('unoscillatingflux.txt') #200 entries, one per bin

# FOR UNIVARIATE RESULTS

univariate = np.load('Univariate_results.npy')

theta23 =  univariate [0][-1]
deltam2 =  univariate [1][-1]
alpha =  univariate [2][-1]
E=np.arange(1e-9,10.05, 0.05)[1:]
L=295

def P_mu (E, L, theta23, deltam2):
    """
    Returns the probability of a neutrino not oscillating, given E, L and oscillation parameters
    P of muon neutrino staying as a muon neutrino
    """
    return 1-((np.sin(2*theta23))**2)*(np.sin((1.267*deltam2*L)/E))**2


#NLL FUNCTION

def NLL_alpha (theta23,deltam2, alpha  =  1): 
    """
    Now including alpha
    Returns NLL given a theta23 and deltam2 and alpha
    Lok at the equation, here alpha is the RATE OF CHANGE
    When alpha  =  1 it's the same so can use this in general 
    """
    lambd_new   =   P_mu (E, L, theta23, deltam2) * alpha *unoscillated  
    mi  =  data
    sum   =   0
    for i in range (len(lambd_new)): #sum over bins starting from bin 1
        sum +=   lambd_new[i]-mi[i]*np.log(lambd_new[i])
    return sum

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
  step = 1e-5
  derivative = (function(theta23,deltam2,alpha+step)-function(theta23, deltam2, alpha-step))/(2*step)
  return derivative

curv_matrix = np.zeros((2,2))

wrt_theta23 = central_diff_theta23(NLL_alpha,theta23, deltam2, alpha)
wrt_theta23_2 = central_diff_theta23(NLL_alpha,wrt_theta23 , deltam2, alpha)

wrt_deltam2 = central_diff_deltam2(NLL_alpha,theta23, deltam2, alpha)
wrt_deltam2_2 = central_diff_deltam2(NLL_alpha, theta23 , wrt_deltam2, alpha)

wrt_theta23_deltam2 = central_diff_deltam2(NLL_alpha, wrt_theta23, deltam2, alpha)

curv_matrix[0,0]=wrt_theta23_2 
curv_matrix[0,1]=wrt_theta23_deltam2
curv_matrix[1,0]=wrt_theta23_deltam2
curv_matrix[1,1]=wrt_deltam2_2



# FOR SIMULATED ANNEALING RESULTS

ann = np.load('Simulatedannealing.npy')

theta23_ann =  ann [0][-1]
deltam2_ann =  ann [1][-1]
alpha_ann =  ann [2][-1]

curv_matrix = np.zeros((2,2))

wrt_theta23 = central_diff_theta23(NLL_alpha,theta23, deltam2, alpha)
wrt_theta23_2 = central_diff_theta23(NLL_alpha,wrt_theta23 , deltam2, alpha)

wrt_deltam2 = central_diff_deltam2(NLL_alpha,theta23, deltam2, alpha)
wrt_deltam2_2 = central_diff_deltam2(NLL_alpha, theta23 , wrt_deltam2, alpha)

wrt_theta23_deltam2 = central_diff_deltam2(NLL_alpha, wrt_theta23, deltam2, alpha)

curv_matrix[0,0]=wrt_theta23_2 
curv_matrix[0,1]=wrt_theta23_deltam2
curv_matrix[1,0]=wrt_theta23_deltam2
curv_matrix[1,1]=wrt_deltam2_2
