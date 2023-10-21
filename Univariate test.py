#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:22:00 2022

@author: claraaldegundemanteca
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 13:48:02 2022

@author: claraaldegundemanteca
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import ListedColormap
import random

font = {'family' : 'serif',
        'weight' : 'regular',
        'size'   : 16}

plt.rc('font', **font)


#NLL FUNCTION

"""
Make theta23 - x
deltam2 - y
NLL_alpha - parabolic with minimum at (2,1)
"""

def NLL_alpha (x,y, alpha  =  1): 
    return math.cosh(x/100)+math.cosh(y/100)



x_range = np.linspace(-50,50, 2000)
y_range = np.linspace(-50,50, 2000)
iterations   =   100


# 1D MINIMISATION (PARABOLIC METHOD), for each parameter
    
def parabolic_x (y, iterations, alpha = 1): 
    """
    Minimises NLL wrt x for a given y and a number of iterations
    Prints if convergence criterion met or not
    Returns value of the optimal x and the minimum NLL obtained
    """
    x_points  =  [0.6,0.65, 0.7] #These 3 points always seem to be downhill otwards min
    NLL_points  =  [] #Obtain their correspondent NLL, to fit a parabola
    for i in x_points:
        NLL_points.append(NLL_alpha(i,y, alpha))
    NLL_points  =  np.array(NLL_points)
    
    minimum_x_list  =  [50] #List of optimal x, will compute difference between terms to check convergence
    for j in range (0, iterations):
        NLL_array  =  []
        P2  =  [] #Legendre polynomial
        for i in range(0,len(x_range)):
            term1  =  (x_range[i]-x_points[1])*(x_range[i]-\
            x_points[2])*NLL_points[0]/((x_points[0]-x_points[1])*(x_points[0]-x_points[2]))
            term2  =  (x_range[i]-x_points[0])*(x_range[i]-\
            x_points[2])*NLL_points[1]/((x_points[1]-x_points[0])*(x_points[1]-x_points[2]))
            term3  =  (x_range[i]-x_points[0])*(x_range[i]-\
            x_points[1])*NLL_points[2]/((x_points[2]-x_points[0])*(x_points[2]-x_points[1]))  
            P2.append(term1+term2+term3)
            NLL_array.append(NLL_alpha(x_range[i],y, alpha))
        P2  =  np.array(P2)
        
        ## See how parabola and points change, check algorithm works fine
        # plt.figure()
        # plt.plot(x_range,NLL_array, color='#002395', label='NLL(${\\theta}_{23}$)')
        # plt.plot(x_points, NLL_points, 'x', color='red')
        # plt.plot(x_range,P2, color='#F9A602', label='P2')
        # plt.ylabel('NLL', fontsize=16)
        # plt.xlabel('${\\theta}_{23}$',fontsize=16)
        # plt.ylim(870,920)
        # plt.legend(fontsize=16)
        # plt.grid()
        
        #Finding the minimum of the parabola. Eq 12.6 on the notes
        num  =  (x_points[2]**2-x_points[1]**2)*NLL_points[0]+(x_points[0]**2-x_points[2]**2)*NLL_points[1]+(x_points[1]**2-x_points[0]**2)*NLL_points[2]
        denom  =  (x_points[2]-x_points[1])*NLL_points[0]+(x_points[0]-x_points[2])*NLL_points[1]+(x_points[1]-x_points[0])*NLL_points[2]
        minimum_x  =  abs(0.5*(num/denom))
        minimum_NLL  =  (NLL_alpha(minimum_x, y, alpha)) #Finds NLL at min of parabola
        # Create new set of NLL and x points
        if minimum_NLL < np.max(NLL_points):
            index  =  np.where(NLL_points  ==  np.max(NLL_points))
            NLL_points  =  np.delete(NLL_points, index)
            new_NLL_points  =  np.append(minimum_NLL,NLL_points)
            x_points  =  np.delete(x_points, index)
            new_x_points  =  np.append(minimum_x,x_points)
            
            x_points  =  new_x_points
            NLL_points  =  new_NLL_points       
        minimum_x_list.append(minimum_x)
        diff  =  abs(minimum_x_list[-1]-minimum_x_list[-2])        
        if diff < 1e-9:
            ##For report
            # plt.figure()
            # plt.plot(x_range,NLL_array)
            # plt.plot(x_points,NLL_points, 'x', label  =  'Fitting points')
            # plt.plot(x_range,P2, label  =  'Parabola')
            # plt.title('$\Delta m^2_{23}$   =   %f' %(y))
            # plt.ylabel('NLL')
            # plt.xlabel('$\theta_{23}$')
            # plt.legend()
            # plt.grid()
            break        
    return minimum_x, minimum_NLL 

def parabolic_y (x, iterations, alpha = 1):
    """
    Given a value of x and no of iterations, returns the y and NLL 
    at which minimum is reached following the parabolic method

    """
    y_points  =  [0.0017,0.0018,0.002] #3 points to the left of prediction, need close otherwise wrong curvature
    NLL_points  =  [] #Obtain their correspondent NLL
    for i in y_points:
        NLL_points.append(NLL_alpha(x,i, alpha))
    NLL_points  =  np.array(NLL_points)
    minimum_y_list  =  [50]
    for j in range (0,iterations):
        """
        Encounter some problems if we dont select a decaying region that can be approximated to a parabola. Check graphs first
        Initial guesses have to be faily close to minimum
        Also need guesses close together so a big parabola is not formed
        """
        NLL_  =  []
        P2  =  []         #Legendre pol
        for i in range(0,len(y_range)):
            term1  =  (y_range[i]-y_points[1])*(y_range[i]-y_points[2])*NLL_points[0]/((y_points[0]-y_points[1])*(y_points[0]-y_points[2]))
            term2  =  (y_range[i]-y_points[0])*(y_range[i]-y_points[2])*NLL_points[1]/((y_points[1]-y_points[0])*(y_points[1]-y_points[2]))
            term3  =  (y_range[i]-y_points[0])*(y_range[i]-y_points[1])*NLL_points[2]/((y_points[2]-y_points[0])*(y_points[2]-y_points[1]))  
            P2.append(term1+term2+term3)
            NLL_.append(NLL_alpha(x,y_range[i], alpha))
        P2  =  np.array(P2)
        # #See how parabolas and points change, for report
        # plt.figure()
        # plt.plot(y_range,NLL_)
        # plt.plot(y_points,NLL_points,'x')
        # plt.plot(y_range,P2)
        # plt.title('${\\theta}_{23}$  =   %f' %(x))
        # plt.ylabel('NLL')
        # plt.xlabel('$Deltam^2_{23}$')        
        #Finding the minimum of the parabola. Eq 12.6 on the notes
        num  =  (y_points[2]**2-y_points[1]**2)*NLL_points[0]+(y_points[0]**2-y_points[2]**2)*NLL_points[1]+(y_points[1]**2-y_points[0]**2)*NLL_points[2]
        denom  =  (y_points[2]-y_points[1])*NLL_points[0]+(y_points[0]-y_points[2])*NLL_points[1]+(y_points[1]-y_points[0])*NLL_points[2]
        minimum_y  =  abs(0.5*(num/denom))
        minimum_NLL  =  NLL_alpha(x, minimum_y, alpha) #Finds NLL at min of parabola        
        # Create new set* of NLL and x points
        if minimum_NLL < np.max(NLL_points):
            index  =  np.where(NLL_points  ==  np.max(NLL_points))
            NLL_points  =  np.delete(NLL_points, index)
            new_NLL_points  =  np.append(minimum_NLL,NLL_points)
            y_points  =  np.delete(y_points, index)
            new_y_points  =  np.append(minimum_y,y_points)          
            y_points  =  new_y_points
            NLL_points  =  new_NLL_points
        minimum_y_list.append(minimum_y)
        diff  =  abs(minimum_y_list[-1]-minimum_y_list[-2])     
        if diff < 1e-9:
            ##For report
            # plt.figure()
            # plt.plot(y_range,NLL_)
            # plt.plot(y_points,NLL_points,'x')
            # plt.plot(y_range,P2)
            # plt.title('Parabolic wrt ${\\Delta}m^2_{23}$, ${\\theta}_{23}$  =   %f' %(x))
            # plt.ylabel('NLL')
            # plt.ylim(-200,1000)
            # plt.xlabel('$\Delta m^2_{23}$')
            break
    return minimum_y, minimum_NLL


def univariate_2D (y, iterations): 
    """
    Returns x and y for which NLL is minimum
    First guess y, use parabolic to get x, use the result as
    a guess to apply parabolic minimisation again to y
    """
    #start with an initial y, minimise x
    x_list  =  [50] 
    y_list  =  [50] #add to list when minimising in each direction
    min_y  =  y
    for i in range(0,iterations):        
        #Find optimal delta m2
        x_list.append(parabolic_x (min_y, 1)[0]) #1 iteration, we want to minimise once in one direction
        min_x  =  x_list[-1]
        diff_x  =  abs(x_list[-2]-x_list[-1])       
        #Find optimal y
        y_list.append(parabolic_y (min_x, 1)[0]) 
        min_y  =  y_list[-1]
        diff_y  =  abs(y_list[-2]-y_list[-1])
        # plt.plot(min_x, min_y, 'o', markersize  =  5, color  =  'black', label='Univariate method result')

        #Stopping condition
        if diff_y < 1e-9 and diff_x < 1e-9:
            NLL_array_x  =  []
            for i in x_range:
                NLL_array_x.append(NLL_alpha(i,y))
            NLL_array_y  =  []
            for i in y_range:
                NLL_array_y.append(NLL_alpha(x_list[-1],i))   
    return x_list[-1],y_list[-1]


results=univariate_2D (0.5, 100)
x_univariate=results[0]
y_univariate=results[1]

X, Y  =   np.meshgrid(x_range, y_range)
Z  =  np.zeros((len(x_range),len(x_range)))

for i in range(0, len(x_range)):
    for j in range(0,len(y_range)):
        Z[j][i]=NLL_alpha(x_range[i],y_range[j])

plt.figure()
plt.subplot(2,1,1)
plt.contourf(X, Y,Z, cmap  =  'jet_r')
cbar=plt.colorbar()
plt.plot(x_univariate,y_univariate, 'o', markersize  =  10, color  =  'black', label='Univariate result')
cbar.set_label('cosh x + cosh y', rotation=90)
# plt.xlabel('x', labelpad  =  20, fontsize=16)
plt.ylabel('y', labelpad  =  20, fontsize=16)
plt.legend()
# plt.savefig('Univariate_test.png')


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
plt.subplot(2,1,2)
plt.contourf(X, Y,Z, cmap  =  'jet_r')
cbar=plt.colorbar()
plt.plot(results[0],results[1], 'o', markersize  =  10, color  =  'black', label='Simulated annealing result')
cbar.set_label('cosh x + cosh y', rotation=90)
plt.xlabel('x', labelpad  =  20, fontsize=16)
plt.ylabel('y', labelpad  =  20, fontsize=16)
plt.legend()
plt.savefig('Annealing_test.png')


