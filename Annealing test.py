#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:46:02 2022

@author: claraaldegundemanteca
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import ListedColormap
font = {'family' : 'serif',
        'weight' : 'regular',
        'size'   : 16}
plt.rc('font', **font)

Kb=1.38e-23
x_range = np.linspace(-50,50, 2000)
y_range = np.linspace(-50,50, 2000)

def NLL_alpha (x,y, alpha  =  1): 
    return math.cosh(x/100)+math.cosh(y/100)

def sim_anneal_alpha(function, x_initial, y_initial, no_cycles):
    """
    Generalised simulated annealing algorithm, including alpha
    Returns optimal parameters and errors given a set of initial parameters
    Have to specify the number of cycles
    If plot = True, all the results from each cycle plot onto a contour of NLL
    If convergence_test = True, it plots the result of each parameter and NLL
    as a function of cycle (check that is stabilises)
    """
    T_0=1
    T_f=1e-8
    cooling_coeff=(T_f/T_0)**(1/(no_cycles-1)) #Fraction of Tf/To: multiply times previous T, cools system
    repetitions= 10 #Per T
    
    #Set upper limits to data to avoid perturbations crossing unpghysical limits
    x_lim=np.pi/2
    y_lim=2e-2
    step=0.01 #between 0 and 1
    perturbation_x=x_lim*step
    perturbation_y=y_lim*step #so we dont go beyond limits
    
    #Initial situation 
    x_i=x_initial
    y_i=y_initial
    E_i=function(x_i, y_i)
    T=T_0
    n_perturbation=0

    #After each cycle, x and y will change, create an empty array for this
    x_array=np.zeros(no_cycles + 1)
    x_array[0]=x_i
    y_array=np.zeros(no_cycles + 1)
    y_array[0]=y_i
    E_array=np.zeros(no_cycles+1) #create an empty array to update it with each cycle
    E_array[0]=E_i

    for i in range(0,no_cycles):
        for j in range (repetitions): #to make sure we add something
            lim1 = perturbation_x*(no_cycles-i)/(no_cycles)
            lim2 = perturbation_y*(no_cycles-i)/(no_cycles)
            x_f=x_i + random.uniform(-lim1,lim1)
            x_f=max(min(x_f, x_lim), 0)
            y_f=y_i + random.uniform(-lim2,lim2)
            y_f=max(min(y_f, y_lim), 0)
            
            E_i=function(x_i, y_i)
            E_f=function(x_f, y_f)
            delta_E=E_f-E_i
            
            if delta_E > 0:
                P_acc=np.exp(-np.abs(delta_E)/(Kb*T))
                if P_acc > random.random(): #greater than a random no between 0 and 1
                        acc = True
                else:
                    acc = False
                    
            else:  #Decrease in energy
                acc = True
            if acc == True:
                #Make the initial the final to start again
                x_i=x_f
                y_i=y_f
                E_i=E_f
                n_perturbation+=1
        #For each cycle, update our array. Save whatever was been redefined as initial: has been accepted  and could go through another repetition
        x_array[i+1] = x_i 
        y_array[i+1] = y_i 
        E_array[i+1] = E_i
    
        #Once the cycle is done, reduce the temperature
        T*=cooling_coeff    

    return [x_i, y_i]

results=sim_anneal_alpha(NLL_alpha, 0.5, 1.5, 500)

X, Y  =   np.meshgrid(x_range, y_range)
Z  =  np.zeros((len(x_range),len(x_range)))

for i in range(0, len(x_range)):
    for j in range(0,len(y_range)):
        Z[j][i]=NLL_alpha(x_range[i],y_range[j])

plt.figure()
plt.contourf(X, Y,Z, cmap  =  'jet_r')
cbar=plt.colorbar()
plt.plot(results[0],results[1], 'o', markersize  =  10, color  =  'black', label='Simulated annealing result')
cbar.set_label('cosh x + cosh y', rotation=90)
plt.xlabel('$x$', labelpad  =  20, fontsize=16)
plt.ylabel('$y$', labelpad  =  20, fontsize=16)
plt.legend()
plt.savefig('Annealing_test.png')



