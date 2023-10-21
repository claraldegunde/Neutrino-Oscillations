#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 21:47:25 2022

@author: claraaldegundemanteca
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
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
E = np.arange(1e-9,10.05, 0.05)[1:]#careful with initial E=0, might give error
binwidth = (E[-1]-E[0])/n_bins

def plot_data_and_simulation (recorded, simulation):
  """
  Given data files
  Plots the recorded data and the simulated unoscillated neutrinos as histograms
  """
  plt.figure()
  plt.bar(E, simulation, align = 'edge', width = binwidth,color = '#002395',alpha = 0.9, label = 'Unoscillated simulation') 
  plt.bar(E, recorded, align = 'edge', width = binwidth, color = '#F9A602',alpha = 1, label = 'Recorded data', linewidth = 0.5) 
  plt.xlabel('Energy (GeV)',fontsize = 14)
  plt.ylabel('Number of entries', fontsize = 14)
  plt.legend(fontsize = 14)
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

theta23_range = np.linspace(1e-9,np.pi/2, 200)
deltam2_range = np.linspace(1e-9, 2e-2, 200)
alpha_range = np.linspace(1e-9, 3, 200)
L  =  295
Kb=1.38e-23



# NEED ERROR FUNCTIONS 

def sigma_theta23 (minimum_theta23, deltam2, alpha  =  1):
    """
    Returns the standard deviation (0.5 change in NLL) for a theta23 result using
    t parabolic minimisation 
    Also needs the deltam23  and alpha at which itll be evaluated
    """
    NLL_array  =  []
    for i in theta23_range:
        NLL_array.append(NLL_alpha(i,deltam2))
    step   =   0.00001
    for i in range(0,500000): #accuracy of this will depend on the number of points on theta23_array, want a good resolution        
        theta23_plus   =   minimum_theta23 + i*step #from minimum then move to right
        NLL_plus   =   NLL_alpha( theta23_plus ,deltam2, alpha)
        NLL_deviation_plus   =   abs(NLL_alpha(minimum_theta23,deltam2, alpha)-NLL_plus) #deviation from minimum
        if NLL_deviation_plus > 0.5: #return the first that gives a change greater than 0.5, small step to ensure its not too off 0.5
            break
    for i in range(0,50000): #accuracy of this will depend on the number of points on theta23_array, want a good resolution        
        theta23_minus   =    minimum_theta23 - i*step #from minimum then move to right
        NLL_minus   =   NLL_alpha(theta23_minus,deltam2, alpha)
        NLL_deviation_minus   =   abs(NLL_alpha(minimum_theta23,deltam2, alpha)-NLL_minus) #deviation from minimum
        if NLL_deviation_minus > 0.5: #return the first that gives a change greater than 0.5, small step to ensure its not too off 0.5
            #Add theta23 minus 
            break   
    sigma  =    theta23_plus - theta23_minus
    return sigma

def sigma_deltam2 (theta23, minimum_deltam2, alpha  =  1):
    """
    Returns sigma on deltam2  given the result of minimisation and the 
    theta23 and alpha at which it was obtained
    """
    NLL_array  =  []
    for i in deltam2_range:
        NLL_array.append(NLL_alpha(theta23,i))
    step   =   0.00001
    for i in range(0,500000): #accuracy of this will depend on the number of points on theta23_array, want a good resolution        
        deltam2_plus   =   minimum_deltam2 + i*step #from minimum then move to right
        NLL_plus   =   NLL_alpha( theta23 ,deltam2_plus, alpha)
        NLL_deviation_plus   =   abs(NLL_alpha(theta23,minimum_deltam2, alpha)-NLL_plus) #deviation from minimum
        if NLL_deviation_plus > 0.5: #return the first that gives a change greater than 0.5, small step to ensure its not too off 0.5
            break
    for i in range(0,500000): #accuracy of this will depend on the number of points on theta23_array, want a good resolution        
        deltam2_minus   =   minimum_deltam2 - i*step #from minimum then move to right
        NLL_minus   =   NLL_alpha( theta23 ,deltam2_minus, alpha)
        NLL_deviation_minus   =   abs(NLL_alpha(theta23,minimum_deltam2, alpha)-NLL_minus) #deviation from minimum
        if NLL_deviation_minus > 0.5: #return the first that gives a change greater than 0.5, small step to ensure its not too off 0.5
            break
    sigma  =    deltam2_plus - deltam2_minus
    return sigma

def sigma_alpha(theta23, deltam2, minimum_alpha):
    """
    Returns sigma on alpha given the result of minimisation and the 
    deltam2 and alpha at which it was obtained
    """
    NLL_array  =  []
    for i in alpha_range:
        NLL_array.append(NLL_alpha(theta23, deltam2, i))
    step   =   0.00001
    for i in range(0,500000): #accuracy of this will depend on the number of points on theta23_array, want a good resolution        
        alpha_plus   =   minimum_alpha + i*step #from minimum then move to right
        NLL_plus   =   NLL_alpha( theta23 ,deltam2, alpha_plus)
        NLL_deviation_plus   =   abs(NLL_alpha(theta23,deltam2, minimum_alpha)-NLL_plus) #deviation from minimum
        if NLL_deviation_plus > 0.5: #return the first that gives a change greater than 0.5, small step to ensure its not too off 0.5
            break
    for i in range(0,500000): #accuracy of this will depend on the number of points on theta23_array, want a good resolution        
        alpha_minus   =   minimum_alpha - i*step #from minimum then move to right
        NLL_minus   =   NLL_alpha( theta23 ,deltam2, alpha_minus)
        NLL_deviation_minus   =   abs(NLL_alpha(theta23,deltam2, minimum_alpha)-NLL_minus) #deviation from minimum
        if NLL_deviation_minus > 0.5: #return the first that gives a change greater than 0.5, small step to ensure its not too off 0.5
            break
    sigma  =    alpha_plus - alpha_minus
    return sigma

def sim_anneal_alpha(function, theta23_initial, deltam2_initial, alpha_initial, no_cycles, plot = True, convergence_test = True):
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
    theta23_lim=np.pi/2
    deltam2_lim=2e-2
    alpha_lim=3
    step=0.01 #between 0 and 1
    perturbation_theta23=theta23_lim*step
    perturbation_deltam2=deltam2_lim*step #so we dont go beyond limits
    perturbation_alpha=alpha_lim*step
    
    #Initial situation 
    theta23_i=theta23_initial
    deltam2_i=deltam2_initial
    alpha_i=deltam2_initial
    E_i=function(theta23_i, deltam2_i,alpha_i )
    T=T_0
    n_perturbation=0

    #After each cycle, theta23 and deltam2 will change, create an empty array for this
    theta23_array=np.zeros(no_cycles + 1)
    theta23_array[0]=theta23_i
    deltam2_array=np.zeros(no_cycles + 1)
    deltam2_array[0]=deltam2_i
    alpha_array=np.zeros(no_cycles + 1)
    alpha_array[0]=alpha_i
    E_array=np.zeros(no_cycles+1) #create an empty array to update it with each cycle
    E_array[0]=E_i
    
    if plot == True:
        X, Y= np.meshgrid(theta23_range, deltam2_range)
        Z=np.zeros((200,200))
        for i in range(0, len(theta23_range)):
            for j in range(0,len(deltam2_range)):
                Z[j][i]=NLL_alpha(theta23_range[i],deltam2_range[j])
        plt.figure()
        plt.contourf(X, Y,Z, cmap='jet_r')
        cbar=plt.colorbar()
        cbar.set_label('NLL', rotation=90)
        plt.xlabel('${\\theta}_{23}$', labelpad=20,**font, fontsize=16)
        plt.ylabel('${\\Delta}m^2_{23}$', labelpad=20, **font, fontsize=16)
        
    for i in range(0,no_cycles):
        for j in range (repetitions): #to make sure we add something
            lim1 = perturbation_theta23*(no_cycles-i)/(no_cycles)
            lim2 = perturbation_deltam2*(no_cycles-i)/(no_cycles)
            lim3 = perturbation_alpha*(no_cycles-i)/(no_cycles)
            theta23_f=theta23_i + random.uniform(-lim1,lim1)
            theta23_f=max(min(theta23_f, theta23_lim), 0)
            deltam2_f=deltam2_i + random.uniform(-lim2,lim2)
            deltam2_f=max(min(deltam2_f, deltam2_lim), 0)
            alpha_f=alpha_i + random.uniform(-lim3,lim3)
            alpha_f=max(min(alpha_f, alpha_lim), 0)
            
            E_i=function(theta23_i, deltam2_i, alpha_i)
            E_f=function(theta23_f, deltam2_f, alpha_f)
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
                theta23_i=theta23_f
                deltam2_i=deltam2_f
                alpha_i=alpha_f
                E_i=E_f
                n_perturbation+=1
        #For each cycle, update our array. Save whatever was been redefined as initial: has been accepted  and could go through another repetition
        theta23_array[i+1] = theta23_i 
        deltam2_array[i+1] = deltam2_i 
        alpha_array[i+1] = alpha_i 

        E_array[i+1] = E_i
    
        #Once the cycle is done, reduce the temperature
        T*=cooling_coeff    
    
        if plot == True:
            plt.scatter(theta23_i , deltam2_i, c='k', edgecolors='white')

    #Calculate errors using the sigma function
    error_theta23 = sigma_theta23 ( theta23_i, deltam2_i, alpha_i )
    error_deltam2 = sigma_deltam2 ( theta23_i, deltam2_i, alpha_i )
    error_alpha = sigma_alpha ( theta23_i, deltam2_i, alpha_i )
    plt.scatter(theta23_i , deltam2_i, c='k', edgecolors='white', label='Simulated annealing')
    plt.legend()
    plt.savefig('Contour_annealing_%f_cycles.png' %(no_cycles))
    
    if convergence_test == True:       
        plt.figure()
        cycles_axis=np.linspace(0,no_cycles+1, no_cycles+1)
        plt.plot(cycles_axis, theta23_array, color='#002395')
        plt.ylabel('${\\theta}_{23}$',fontsize=16)
        plt.xlabel('Number of cycle',fontsize=16)
        plt.grid()
        # plt.savefig('Convergence_ann_4.png')
        
        # plt.subplot(2,2,2)
        plt.figure()
        plt.plot(cycles_axis, deltam2_array, color='#002395')
        plt.ylabel('${\\Delta}m^2_{23}$',fontsize=16)
        plt.xlabel('Number of cycle',fontsize=16)
        plt.grid()
        plt.savefig('Convergence_ann_3.png')
        
        # ax3.subplot(2,2,3)
        plt.figure()
        plt.plot(cycles_axis, alpha_array, color='#002395')
        plt.ylabel('${\\alpha}$',fontsize=16)
        plt.xlabel('Number of cycle',fontsize=16)
        plt.grid()
        # plt.savefig('Convergence_ann_2.png')

        # plt.subplot(2,2,4)
        plt.figure()
        plt.plot(cycles_axis, E_array, color='#002395')
        plt.ylabel('NLL', fontsize=16)
        plt.xlabel('Number of cycle', fontsize=16)
        plt.grid()
        # plt.savefig('Convergence_ann_1.png')
    return [theta23_i, deltam2_i, alpha_i, error_theta23, error_deltam2, error_alpha]
            

parameters_and_errors=sim_anneal_alpha(NLL_alpha, 0.7, 2e-3, 2,5000, plot=True, convergence_test=False)
theta23_sim_an_alpha=parameters_and_errors[0]
deltam2_sim_an_alpha=parameters_and_errors[1]
alpha_sim_an_alpha=parameters_and_errors[2]
err_theta23_sim_an_alpha=parameters_and_errors[3]
err_deltam2_sim_an_alpha=parameters_and_errors[4]
err_alpha_sim_an_alpha=parameters_and_errors[5]
np.save('Simulatedannealing',parameters_and_errors)


#PRINT RESULTS

print('From simulated annealing method')
print('theta23 = ', theta23_sim_an_alpha, '+-', err_theta23_sim_an_alpha)
print('deltam2 = ', deltam2_sim_an_alpha, '+-', err_deltam2_sim_an_alpha)
print('alpha = ', alpha_sim_an_alpha, '+-',err_alpha_sim_an_alpha )
 


#COMPARE WITH RECORDED DATA AND WITH UNIVARIATE
E=np.linspace(1e-9,10,200)
P_parameters = P_mu (E, L, theta23_sim_an_alpha, deltam2_sim_an_alpha)
modified_unoscillated=unoscillated*P_parameters*alpha_sim_an_alpha*E

plt.figure()
plt.subplot(1,2,1)
plt.bar(E, data, align='edge', width=binwidth, color='#F9A602',alpha=1, label='Recorded data') 
plt.hist(E, bins=E, weights=modified_unoscillated, color='black', histtype='step', label='Simmulated annealing')
plt.xlabel('Energy (GeV)')
plt.ylabel('Number of entries')
plt.grid(alpha=0.7)
plt.legend()


P_parameters = P_mu (E, L, 0.7545722964679525, 0.0022934888739111043)
modified_unoscillated=unoscillated*P_parameters*1.2938869321588002*E
plt.subplot(1,2,2)
plt.hist(E, bins=E, weights=modified_unoscillated, color='black', histtype='step', label='Univariate')
plt.bar(E, data, align='edge', width=binwidth, color='#F9A602',alpha=1, label='Recorded data') 
plt.xlabel('Energy (GeV)')
# plt.ylabel('Number of entries', fontsize= 12)
plt.legend()
plt.grid(alpha=0.7)
plt.savefig('Comparison_methods.png')


#NOT TALINKG ALPHA INTO ACCOUNT ALPHA

Kb=1.38e-23

def sim_anneal(function, theta23_initial, deltam2_initial, no_cycles, plot = True, convergence_test = True):
    T_0=1
    T_f=1e-8
    cooling_coeff=(T_f/T_0)**(1/(no_cycles-1)) #Fraction of Tf/To: multiply times previous T, cools system
    repetitions= 10 #Per T
    
    #Set upper limits to data to avoid perturbations crossing unpghysical limits
    theta23_lim=np.pi/2
    deltam2_lim=6e-3
    step=0.01 #between 0 and 1
    perturbation_theta23=theta23_lim*step
    perturbation_deltam2=deltam2_lim*step #so we dont go beyond limits
    
    #Initial situation 
    theta23_i=theta23_initial
    deltam2_i=deltam2_initial
    E_i=function(theta23_i, deltam2_i)
    T=T_0
    n_perturbation=0

    #After each cycle, theta23 and deltam2 will change, create an empty array for this
    theta23_array=np.zeros(no_cycles + 1)
    theta23_array[0]=theta23_i
    deltam2_array=np.zeros(no_cycles + 1)
    deltam2_array[0]=deltam2_i
    E_array=np.zeros(no_cycles+1) #create an empty array to update it with each cycle
    E_array[0]=E_i
    
    if plot == True:
        # fig1, ax1 = plt.subplots()
        X, Y= np.meshgrid(theta23_range, deltam2_range)
        Z=np.zeros((200,200))
        for i in range(0, len(theta23_range)):
            for j in range(0,len(deltam2_range)):
                Z[j][i]=NLL_alpha(theta23_range[i],deltam2_range[j])
        plt.figure()
        plt.contourf(X, Y,Z, cmap='cividis')
        plt.colorbar()
        plt.xlabel('${\\theta}_{23}$', labelpad=20)
        plt.ylabel('${\\Delta}m^2_{23}$', labelpad=20)
        plt.title('Contour plot')
 
    for i in range(0,no_cycles):
        for j in range (repetitions): #to make sure we add something
            lim1 = perturbation_theta23*(no_cycles-i)/(no_cycles)
            lim2 = perturbation_theta23*(no_cycles-i)/(no_cycles)
            theta23_f=theta23_i + random.uniform(-lim1,lim1)
            theta23_f=max(min(theta23_f, theta23_lim), 0)
            deltam2_f=deltam2_i + random.uniform(-lim2,lim2)
            deltam2_f=max(min(deltam2_f, deltam2_lim), 0)
            
            E_i=function(theta23_i, deltam2_i)
            E_f=function(theta23_f, deltam2_f)
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
                theta23_i=theta23_f
                deltam2_i=deltam2_f
                E_i=E_f
                n_perturbation+=1
        #For each cycle, update our array. Save whatever was been redefined as initial: has been accepted  and could go through another repetition
        theta23_array[i+1]=theta23_i 
        deltam2_array[i+1]=deltam2_i 
        E_array[i+1]=E_i
    
        #Once the cycle is done, reduce the energy
        T*=cooling_coeff    
    
        if plot == True:
            plt.plot(theta23_i , deltam2_i , 'o', color='red',markersize=2.5)
    
    if convergence_test == True:
        # cycles_axis=np.linspace(0,no_cycles+1, no_cycles+1)
        # plt.figure()
        # plt.plot(cycles_axis, theta23_array)
        # plt.ylabel('${\\theta}_{23}$')
        # plt.xlabel('Number of cycle')
        # plt.title(('${\\theta}_{23}$ convergence for %i cycles') %(no_cycles))
        # plt.grid()
        
        # cycles_axis=np.linspace(0,no_cycles+1, no_cycles+1)
        # plt.figure()
        # plt.plot(cycles_axis, deltam2_array)
        # plt.ylabel('${\\Delta}m^2_{23}$')
        # plt.xlabel('Number of cycle')
        # plt.title(('${\\Delta}m^2_{23}$ convergence for %i cycles') %(no_cycles))
        # plt.grid()
        
        cycles_axis=np.linspace(0,no_cycles+1, no_cycles+1)
        plt.figure()
        plt.plot(cycles_axis, E_array)
        plt.ylabel('NLL_alpha')
        plt.xlabel('Number of cycle')
        plt.title(('NLL_alpha convergence for %i cycles') %(no_cycles))
        plt.grid()
    return theta23_i, deltam2_i

#Print results
theta23_sim_an=sim_anneal(NLL_alpha, np.pi/4, 2e-3, 800, plot=True, convergence_test=False)[0]
deltam2_sim_an=sim_anneal(NLL_alpha, np.pi/4, 2e-3, 800, plot=False, convergence_test=False)[1]
err_theta23_sim_an= sigma_theta23(theta23_sim_an, deltam2_sim_an, 1 )
err_deltam2_sim_an= sigma_deltam2(theta23_sim_an, deltam2_sim_an,1)
print('From simulated annealing method')
print('theta23 = ', theta23_sim_an, '+-', err_theta23_sim_an)
print('deltam2 = ', deltam2_sim_an, '+-', err_deltam2_sim_an)
 

#COMPARE WITH RECORDED DATA AND WITH UNIVARIATE
E=np.linspace(1e-9,10,200)
P_parameters = P_mu (E, L, theta23_sim_an_alpha, deltam2_sim_an_alpha)
modified_unoscillated=unoscillated*P_parameters

plt.figure()
plt.subplot(1,2,1)
plt.bar(E, data, align='edge', width=binwidth, color='#F9A602',alpha=1, label='Recorded data') 
plt.hist(E, bins=E, weights=modified_unoscillated, color='black', histtype='step', label='Simmulated annealing')
plt.xlabel('Energy (GeV)')
plt.ylabel('Number of entries')
plt.grid(alpha=0.7)
plt.legend()


P_parameters = P_mu (E, L, 0.7236613242514341, 0.002593681432511292)
modified_unoscillated=unoscillated*P_parameters
plt.subplot(1,2,2)
plt.hist(E, bins=E, weights=modified_unoscillated, color='black', histtype='step', label='Univariate')
plt.bar(E, data, align='edge', width=binwidth, color='#F9A602',alpha=1, label='Recorded data') 
plt.xlabel('Energy (GeV)')
# plt.ylabel('Number of entries', fontsize= 12)
plt.legend()
plt.grid(alpha=0.7)
plt.savefig('Comparison_methods.png')

    

