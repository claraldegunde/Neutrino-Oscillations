#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 13:48:02 2022

@author: claraaldegundemanteca
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
font = {'family' : 'serif',
        'weight' : 'regular',
        'size'   : 16}

plt.rc('font', **font)


#DATA

# Obseved, experimental data 
data   =   np.loadtxt('data.txt', delimiter  =  '\t', skiprows  =  0) #200 entries, one per bin

# Simmulation, if there was no neutrino oscillations 
unoscillated   =   np.loadtxt('unoscillatingflux.txt') #200 entries, one per bin

# Set energy scale 
n_bins   =   200 #200 for a binwidth of 0.05
E  =  np.arange(1e-9,10.05, 0.05)[1:]
binwidth   =   (E[-1]-E[0])/n_bins

def plot_data_and_simulation (recorded, simulation):
    """
    Given data files
    Plots the recorded data and the simulated unoscillated neutrinos as histograms
    """
    plt.figure()
    plt.bar(E, simulation, align  =  'edge', width  =  binwidth,color  =  '#002395',alpha  =  0.9, label  =  'Unoscillated simulation') 
    plt.bar(E, recorded, align  =  'edge', width  =  binwidth, color  =  '#F9A602',alpha  =  1, label  =  'Recorded data', linewidth  =  0.5) 
    plt.xlabel('Energy (GeV)',fontsize  =  14)
    plt.ylabel('Number of entries', fontsize  =  14)
    plt.legend(fontsize  =  14)
    plt.grid(alpha  =  0.7, linewidth  =  0.3)
    plt.savefig('Data and sim.png')
       
    

#PDF DISTRIBUTION FOR UNOSCILLATED

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

theta23_range = np.linspace(1e-9,np.pi/2, 200)
deltam2_range = np.linspace(1e-9, 2e-2, 200)
alpha_range  =  np.linspace(1e-9, 3, 200)
L   =   295
iterations   =   100
deltam2   =   2.4e-4



# 1D MINIMISATION (PARABOLIC METHOD), for each parameter
    
def parabolic_theta23 (deltam2, iterations, alpha = 1): 
    """
    Minimises NLL wrt theta23 for a given deltam2 and a number of iterations
    Prints if convergence criterion met or not
    Returns value of the optimal theta23 and the minimum NLL obtained
    """
    theta23_points  =  [0.6,0.65, 0.7] #These 3 points always seem to be downhill otwards min
    NLL_points  =  [] #Obtain their correspondent NLL, to fit a parabola
    for i in theta23_points:
        NLL_points.append(NLL_alpha(i,deltam2, alpha))
    NLL_points  =  np.array(NLL_points)
    
    minimum_theta23_list  =  [50] #List of optimal theta23, will compute difference between terms to check convergence
    for j in range (0, iterations):
        NLL_array  =  []
        P2  =  [] #Legendre polynomial
        for i in range(0,len(theta23_range)):
            term1  =  (theta23_range[i]-theta23_points[1])*(theta23_range[i]-\
            theta23_points[2])*NLL_points[0]/((theta23_points[0]-theta23_points[1])*(theta23_points[0]-theta23_points[2]))
            term2  =  (theta23_range[i]-theta23_points[0])*(theta23_range[i]-\
            theta23_points[2])*NLL_points[1]/((theta23_points[1]-theta23_points[0])*(theta23_points[1]-theta23_points[2]))
            term3  =  (theta23_range[i]-theta23_points[0])*(theta23_range[i]-\
            theta23_points[1])*NLL_points[2]/((theta23_points[2]-theta23_points[0])*(theta23_points[2]-theta23_points[1]))  
            P2.append(term1+term2+term3)
            NLL_array.append(NLL_alpha(theta23_range[i],deltam2, alpha))
        P2  =  np.array(P2)
        
        ## See how parabola and points change, check algorithm works fine
        # plt.figure()
        # plt.plot(theta23_range,NLL_array, color='#002395', label='NLL(${\\theta}_{23}$)')
        # plt.plot(theta23_points, NLL_points, 'x', color='red')
        # plt.plot(theta23_range,P2, color='#F9A602', label='P2')
        # plt.ylabel('NLL', fontsize=16)
        # plt.xlabel('${\\theta}_{23}$',fontsize=16)
        # plt.ylim(870,920)
        # plt.legend(fontsize=16)
        # plt.grid()
        
        #Finding the minimum of the parabola. Eq 12.6 on the notes
        num  =  (theta23_points[2]**2-theta23_points[1]**2)*NLL_points[0]+(theta23_points[0]**2-theta23_points[2]**2)*NLL_points[1]+(theta23_points[1]**2-theta23_points[0]**2)*NLL_points[2]
        denom  =  (theta23_points[2]-theta23_points[1])*NLL_points[0]+(theta23_points[0]-theta23_points[2])*NLL_points[1]+(theta23_points[1]-theta23_points[0])*NLL_points[2]
        minimum_theta23  =  abs(0.5*(num/denom))
        minimum_NLL  =  (NLL_alpha(minimum_theta23, deltam2, alpha)) #Finds NLL at min of parabola
        # Create new set of NLL and theta23 points
        if minimum_NLL < np.max(NLL_points):
            index  =  np.where(NLL_points  ==  np.max(NLL_points))
            NLL_points  =  np.delete(NLL_points, index)
            new_NLL_points  =  np.append(minimum_NLL,NLL_points)
            theta23_points  =  np.delete(theta23_points, index)
            new_theta23_points  =  np.append(minimum_theta23,theta23_points)
            
            theta23_points  =  new_theta23_points
            NLL_points  =  new_NLL_points       
        minimum_theta23_list.append(minimum_theta23)
        diff  =  abs(minimum_theta23_list[-1]-minimum_theta23_list[-2])        
        if diff < 1e-9:
            ##For report
            # plt.figure()
            # plt.plot(theta23_range,NLL_array)
            # plt.plot(theta23_points,NLL_points, 'x', label  =  'Fitting points')
            # plt.plot(theta23_range,P2, label  =  'Parabola')
            # plt.title('$\Delta m^2_{23}$   =   %f' %(deltam2))
            # plt.ylabel('NLL')
            # plt.xlabel('$\theta_{23}$')
            # plt.legend()
            # plt.grid()
            break        
    return minimum_theta23, minimum_NLL 

parabolic_minimum_theta23  =   parabolic_theta23 (deltam2, iterations) [0]        
parabolic_minimum_NLL  =   parabolic_theta23 (deltam2,iterations) [1]             

"""
Checked it worked for different deltam2
"""

def parabolic_deltam2 (theta23, iterations, alpha = 1):
    """
    Given a value of theta23 and no of iterations, returns the deltam2 and NLL 
    at which minimum is reached following the parabolic method

    """
    deltam2_points  =  [0.0030,0.0027,0.0025] #3 points to the left of prediction, need close otherwise wrong curvature
    NLL_points  =  [] #Obtain their correspondent NLL
    for i in deltam2_points:
        NLL_points.append(NLL_alpha(theta23,i, alpha))
    NLL_points  =  np.array(NLL_points)
    minimum_deltam2_list  =  [50]
    for j in range (0,iterations):
        """
        Encounter some problems if we dont select a decaying region that can be approximated to a parabola. Check graphs first
        Initial guesses have to be faily close to minimum
        Also need guesses close together so a big parabola is not formed
        """
        NLL_  =  []
        P2  =  []         #Legendre pol
        for i in range(0,len(deltam2_range)):
            term1  =  (deltam2_range[i]-deltam2_points[1])*(deltam2_range[i]-deltam2_points[2])*NLL_points[0]/((deltam2_points[0]-deltam2_points[1])*(deltam2_points[0]-deltam2_points[2]))
            term2  =  (deltam2_range[i]-deltam2_points[0])*(deltam2_range[i]-deltam2_points[2])*NLL_points[1]/((deltam2_points[1]-deltam2_points[0])*(deltam2_points[1]-deltam2_points[2]))
            term3  =  (deltam2_range[i]-deltam2_points[0])*(deltam2_range[i]-deltam2_points[1])*NLL_points[2]/((deltam2_points[2]-deltam2_points[0])*(deltam2_points[2]-deltam2_points[1]))  
            P2.append(term1+term2+term3)
            NLL_.append(NLL_alpha(theta23,deltam2_range[i], alpha))
        P2  =  np.array(P2)
        # #See how parabolas and points change, for report
        # plt.figure()
        # plt.plot(deltam2_range,NLL_)
        # plt.plot(deltam2_points,NLL_points,'x')
        # plt.plot(deltam2_range,P2)
        # plt.title('${\\theta}_{23}$  =   %f' %(theta23))
        # plt.ylabel('NLL')
        # plt.ylim(904,914)
        # plt.xlim(0,4e-3)
        plt.xlabel('$Deltam^2_{23}$')        
        #Finding the minimum of the parabola. Eq 12.6 on the notes
        num  =  (deltam2_points[2]**2-deltam2_points[1]**2)*NLL_points[0]+(deltam2_points[0]**2-deltam2_points[2]**2)*NLL_points[1]+(deltam2_points[1]**2-deltam2_points[0]**2)*NLL_points[2]
        denom  =  (deltam2_points[2]-deltam2_points[1])*NLL_points[0]+(deltam2_points[0]-deltam2_points[2])*NLL_points[1]+(deltam2_points[1]-deltam2_points[0])*NLL_points[2]
        minimum_deltam2  =  abs(0.5*(num/denom))
        minimum_NLL  =  NLL_alpha(theta23, minimum_deltam2, alpha) #Finds NLL at min of parabola        
        # Create new set* of NLL and theta23 points
        if minimum_NLL < np.max(NLL_points):
            index  =  np.where(NLL_points  ==  np.max(NLL_points))
            NLL_points  =  np.delete(NLL_points, index)
            new_NLL_points  =  np.append(minimum_NLL,NLL_points)
            deltam2_points  =  np.delete(deltam2_points, index)
            new_deltam2_points  =  np.append(minimum_deltam2,deltam2_points)          
            deltam2_points  =  new_deltam2_points
            NLL_points  =  new_NLL_points
        minimum_deltam2_list.append(minimum_deltam2)
        diff  =  abs(minimum_deltam2_list[-1]-minimum_deltam2_list[-2])     
        if diff < 1e-9:
            ##For report
            # plt.figure()
            # plt.plot(deltam2_range,NLL_)
            # plt.plot(deltam2_points,NLL_points,'x')
            # plt.plot(deltam2_range,P2)
            # plt.title('Parabolic wrt ${\\Delta}m^2_{23}$, ${\\theta}_{23}$  =   %f' %(theta23))
            # plt.ylabel('NLL')
            # plt.ylim(-200,1000)
            # plt.xlabel('$\Delta m^2_{23}$')
            break
    return minimum_deltam2, minimum_NLL

def parabolic_alpha (theta23, deltam2, iterations):
    """
    Given a value of theta23 and no of iterations, returns the alpha and NLL 
    at which minimum is reached following the parabolic method

    """
    alpha_points  =  [1.15,1.20,1.25] #3 points to the left of prediction, need close otherwise wrong curvature
    NLL_points  =  [] #Obtain their correspondent NLL
    for i in alpha_points:
        NLL_points.append(NLL_alpha(theta23,deltam2, i))
    NLL_points  =  np.array(NLL_points)
    minimum_alpha_list  =  [50]
    for j in range (0,iterations):
        """
        Encounter some problems if we dont select a decaying region that can be approximated to a parabola. Check graphs first
        Initial guesses have to be faily close to minimum
        Also need guesses close together so a big parabola is not formed
        """
        NLL_  =  []
        P2  =  []         #Legendre pol
        for i in range(0,len(alpha_range)):
            term1  =  (alpha_range[i]-alpha_points[1])*(alpha_range[i]-alpha_points[2])*NLL_points[0]/((alpha_points[0]-alpha_points[1])*(alpha_points[0]-alpha_points[2]))
            term2  =  (alpha_range[i]-alpha_points[0])*(alpha_range[i]-alpha_points[2])*NLL_points[1]/((alpha_points[1]-alpha_points[0])*(alpha_points[1]-alpha_points[2]))
            term3  =  (alpha_range[i]-alpha_points[0])*(alpha_range[i]-alpha_points[1])*NLL_points[2]/((alpha_points[2]-alpha_points[0])*(alpha_points[2]-alpha_points[1]))  
            P2.append(term1+term2+term3)
            NLL_.append(NLL_alpha(theta23,deltam2, alpha_range[i]))
        P2  =  np.array(P2)
        # #See how parabolas and points change, for report
        # plt.figure()
        # plt.plot(alpha_range,NLL_)
        # plt.plot(alpha_points,NLL_points,'x')
        # plt.plot(alpha_range,P2)
        # plt.title('${\\theta}_{23}$  =   %f' %(theta23))
        # plt.ylabel('NLL')
        # plt.xlabel('$Deltam^2_{23}$')        
        #Finding the minimum of the parabola. Eq 12.6 on the notes
        num  =  (alpha_points[2]**2-alpha_points[1]**2)*NLL_points[0]+(alpha_points[0]**2-alpha_points[2]**2)*NLL_points[1]+(alpha_points[1]**2-alpha_points[0]**2)*NLL_points[2]
        denom  =  (alpha_points[2]-alpha_points[1])*NLL_points[0]+(alpha_points[0]-alpha_points[2])*NLL_points[1]+(alpha_points[1]-alpha_points[0])*NLL_points[2]
        minimum_alpha  =  abs(0.5*(num/denom))
        minimum_NLL  =  NLL_alpha(theta23, deltam2, minimum_alpha) #Finds NLL at min of parabola        
        # Create new set* of NLL and theta23 points
        if minimum_NLL < np.max(NLL_points):
            index  =  np.where(NLL_points  ==  np.max(NLL_points))
            NLL_points  =  np.delete(NLL_points, index)
            new_NLL_points  =  np.append(minimum_NLL,NLL_points)
            alpha_points  =  np.delete(alpha_points, index)
            new_alpha_points  =  np.append(minimum_alpha,alpha_points)          
            alpha_points  =  new_alpha_points
            NLL_points  =  new_NLL_points
        minimum_alpha_list.append(minimum_alpha)
        diff  =  abs(minimum_alpha_list[-1]-minimum_alpha_list[-2])     
        if diff < 1e-9:
            print('Convergence reached')
            ##For report
            # plt.figure()
            # plt.plot(alpha_range,NLL_)
            # plt.plot(alpha_points,NLL_points,'x')
            # plt.plot(alpha_range,P2)
            # plt.title('Parabolic wrt ${\\Delta}m^2_{23}$, ${\\theta}_{23}$  =   %f' %(theta23))
            # plt.ylabel('NLL')
            # plt.ylim(-200,1000)
            # plt.xlabel('$\Delta m^2_{23}$')
            break
    return minimum_alpha, minimum_NLL



#ACCURACY OF RESULT FOR PARABOLIC (get sigma, change of 0.5 on NLL for all 3 variables)

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
            #To see how it changes
            # plt.figure()
            # plt.plot(theta23_range, NLL_array)
            # plt.plot(theta23_plus,NLL_plus, 'x', color  =  'red', label  =  'NLL difference 0.5')
            # plt.plot(minimum_theta23,NLL_alpha(minimum_theta23,deltam2, alpha), 'x', color  =  'green', label  =  'Minimum')
            # plt.ylabel('NLL')
            # plt.xlabel('${\\theta_{23}}$')
            # plt.legend()
            break
    for i in range(0,50000): #accuracy of this will depend on the number of points on theta23_array, want a good resolution        
        theta23_minus   =    minimum_theta23 - i*step #from minimum then move to right
        NLL_minus   =   NLL_alpha(theta23_minus,deltam2, alpha)
        NLL_deviation_minus   =   abs(NLL_alpha(minimum_theta23,deltam2, alpha)-NLL_minus) #deviation from minimum
        if NLL_deviation_minus > 0.5: #return the first that gives a change greater than 0.5, small step to ensure its not too off 0.5
            #Add theta23 minus 
            # plt.plot(theta23_minus,NLL_minus, 'x', color  =  'red', label  =  'NLL difference 0.5')
            # plt.grid()
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
            #To see how it changes
            # plt.figure()
            # plt.plot(deltam2_range, NLL_array)
            # plt.plot(deltam2_plus,NLL_plus, 'x', color  =  'red', label  =  'NLL difference 0.5')
            # plt.plot(minimum_deltam2,NLL_alpha(theta23,minimum_deltam2, alpha), 'x', color  =  'green', label  =  'Minimum')
            # plt.ylabel('NLL')
            # plt.xlabel('${\\theta_{23}}$')
            # plt.legend()
            break
    for i in range(0,500000): #accuracy of this will depend on the number of points on theta23_array, want a good resolution        
        deltam2_minus   =   minimum_deltam2 - i*step #from minimum then move to right
        NLL_minus   =   NLL_alpha( theta23 ,deltam2_minus, alpha)
        NLL_deviation_minus   =   abs(NLL_alpha(theta23,minimum_deltam2, alpha)-NLL_minus) #deviation from minimum
        if NLL_deviation_minus > 0.5: #return the first that gives a change greater than 0.5, small step to ensure its not too off 0.5
            # plt.plot(deltam2_minus,NLL_minus, 'x', color  =  'red', label  =  'NLL difference 0.5')
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
            #To see how it changes
            # plt.figure()
            # plt.plot(alpha_range, NLL_array)
            # plt.plot(alpha_plus,NLL_plus, 'x', color  =  'red', label  =  'NLL difference 0.5')
            # plt.plot(minimum_alpha,NLL_alpha(theta23,deltam2, minimum_alpha), 'x', color  =  'green', label  =  'Minimum')
            # plt.ylabel('NLL')
            # plt.xlabel('${\\theta_{23}}$')
            # plt.legend()
            break
    for i in range(0,500000): #accuracy of this will depend on the number of points on theta23_array, want a good resolution        
        alpha_minus   =   minimum_alpha - i*step #from minimum then move to right
        NLL_minus   =   NLL_alpha( theta23 ,deltam2, alpha_minus)
        NLL_deviation_minus   =   abs(NLL_alpha(theta23,deltam2, minimum_alpha)-NLL_minus) #deviation from minimum
        if NLL_deviation_minus > 0.5: #return the first that gives a change greater than 0.5, small step to ensure its not too off 0.5
            #To see how it changes
            plt.plot(alpha_minus,NLL_minus, 'x', color  =  'red', label  =  'NLL difference 0.5')
            break
    sigma  =    alpha_plus - alpha_minus
    return sigma


def univariate_2D (deltam2, iterations): 
    """
    Returns theta23 and deltam2 for which NLL is minimum
    First guess deltam2, use parabolic to get theta23, use the result as
    a guess to apply parabolic minimisation again to deltam2
    """
    #start with an initial deltam2, minimise theta23
    theta23_list  =  [50] 
    deltam2_list  =  [50] #add to list when minimising in each direction
    min_deltam2  =  deltam2
    for i in range(0,iterations):        
        #Find optimal delta m2
        theta23_list.append(parabolic_theta23 (min_deltam2, 1)[0]) #1 iteration, we want to minimise once in one direction
        min_theta23  =  theta23_list[-1]
        diff_theta23  =  abs(theta23_list[-2]-theta23_list[-1])       
        #Find optimal deltam2
        deltam2_list.append(parabolic_deltam2 (min_theta23, 1)[0]) 
        min_deltam2  =  deltam2_list[-1]
        diff_deltam2  =  abs(deltam2_list[-2]-deltam2_list[-1])
        #Stopping condition
        if diff_deltam2 < 1e-9 and diff_theta23 < 1e-9:
            NLL_array_theta23  =  []
            for i in theta23_range:
                NLL_array_theta23.append(NLL_alpha(i,deltam2))
            NLL_array_deltam2  =  []
            for i in deltam2_range:
                NLL_array_deltam2.append(NLL_alpha(theta23_list[-1],i))   
    return theta23_list[-1], deltam2_list[-1]

def univariate_3D (deltam2, alpha, iterations): 
    """
    Returns theta23, deltam2 and alpha for which NLL is minimum
    First guess deltam2, use parabolic to get theta23, use the result as
    a guess to apply parabolic minimisation again to deltam2
    """

    #start with an initial deltam2, minimise theta23
    theta23_list  =  [50] 
    deltam2_list  =  [50] #add to list when minimising in each direction
    alpha_list = [50]
    min_deltam2  =  deltam2
    min_alpha = alpha
    for i in range(0,iterations):        
        #Find optimal delta m2
        theta23_list.append(parabolic_theta23 (min_deltam2, 1, min_alpha)[0]) #1 iteration, we want to minimise once in one direction
        min_theta23  =  theta23_list[-1]
        diff_theta23  =  abs(theta23_list[-2]-theta23_list[-1])       
        #Find optimal deltam2
        deltam2_list.append(parabolic_deltam2 (min_theta23, 1, min_alpha)[0]) 
        min_deltam2  =  deltam2_list[-1]
        diff_deltam2  =  abs(deltam2_list[-2]-deltam2_list[-1])
        #Find optimal alpha
        alpha_list.append(parabolic_alpha (min_theta23, min_deltam2, 1)[0]) 
        min_alpha  =  alpha_list[-1]
        # diff_alpha  =  abs(alpha_list[-2]-alpha_list[-1])
        #Stopping condition
        if diff_deltam2 < 1e-9 and diff_theta23 < 1e-9:
            NLL_array_theta23  =  []
            for i in theta23_range:
                NLL_array_theta23.append(NLL_alpha(i,deltam2))
            NLL_array_deltam2  =  []
            for i in deltam2_range:
                NLL_array_deltam2.append(NLL_alpha(theta23_list[-1],i))   
            break
    return theta23_list,deltam2_list, alpha_list


#RESULTS 

iterations=500
#Plot results
results = univariate_3D(deltam2, 1.5, iterations)
theta23_univariate =  results [0][-1]
deltam2_univariate =  results [1][-1]
alpha_univariate =  results [2][-1]
err_theta23_univariate = sigma_theta23(theta23_univariate, deltam2_univariate, alpha_univariate )
err_deltam2_univariate = sigma_deltam2(theta23_univariate, deltam2_univariate, alpha_univariate )
err_alpha_univariate = sigma_alpha(theta23_univariate, deltam2_univariate, alpha_univariate )
np.save('Univariate_results', results)


#PLOT RESULTS ONTO CONTOUR

X, Y  =   np.meshgrid(theta23_range, deltam2_range)
Z  =  np.zeros((200,200))

for i in range(0, len(theta23_range)):
    for j in range(0,len(deltam2_range)):
        Z[j][i]=NLL_alpha(theta23_range[i],deltam2_range[j])

plt.figure()
plt.contourf(X, Y,Z, cmap  =  'jet_r')
cbar=plt.colorbar()
cbar.set_label('NLL', rotation=90)
# plt.plot(theta23_univariate,deltam2_univariate, 'o', markersize  =  5, color  =  'black', label='Univariate method result')
plt.errorbar(theta23_univariate,deltam2_univariate, yerr=err_deltam2_univariate, xerr=err_theta23_univariate, fmt='o', markersize  =  10, color  =  'black',label='Univariate result')
# plt.scatter(results[0][1:] , results[1][1:], c='k', edgecolors='white', label='Univariate method')
plt.xlabel('${\\theta}_{23}$', labelpad  =  20, fontsize=16)
plt.ylabel('${\\Delta}m^2_{23}$', labelpad  =  20, fontsize=16)
plt.legend()
plt.savefig('Contour_univariate.png')


#PRINT RESULTS
print('From univariate method')
print('theta23 = ', theta23_univariate, '+-', err_theta23_univariate)
print('deltam2 = ', deltam2_univariate, '+-', err_deltam2_univariate)
print('alpha = ', alpha_univariate,  '+-', err_alpha_univariate)



#NOT TAKING ALPHA INTO ACCOUNT

results = univariate_2D(deltam2, iterations)

theta23_univariate =  results [0]
deltam2_univariate =  results [1]
err_theta23_univariate = sigma_theta23(theta23_univariate, deltam2_univariate)
err_deltam2_univariate = sigma_deltam2(theta23_univariate, deltam2_univariate)


# X, Y  =   np.meshgrid(theta23_range, deltam2_range)
# Z  =  np.zeros((200,200))

# for i in range(0, len(theta23_range)):
#     for j in range(0,len(deltam2_range)):
#         Z[j][i]=NLL_alpha(theta23_range[i],deltam2_range[j])

# plt.figure()
# plt.contourf(X, Y,Z, cmap  =  'jet_r')
# cbar=plt.colorbar()
# cbar.set_label('NLL', rotation=90)
# # plt.plot(theta23_univariate,deltam2_univariate, 'o', markersize  =  5, color  =  'black', label='Univariate method result')
# # plt.errorbar(theta23_univariate,deltam2_univariate, yerr=err_deltam2_univariate, xerr=err_theta23_univariate, fmt='o', markersize  =  10, color  =  'black',label='Univariate method result')
# plt.scatter(results[0] , results[1], c='k', edgecolors='white', label='Univariate method')
# plt.xlabel('${\\theta}_{23}$', labelpad  =  20, fontsize=16)
# plt.ylabel('${\\Delta}m^2_{23}$', labelpad  =  20, fontsize=16)
# plt.legend()
# plt.savefig('Contour_univariate.png')



#PRINT RESULTS

print('From univariate method, without alpha')
print('theta23 = ', theta23_univariate, '+-', err_theta23_univariate)
print('deltam2 = ', deltam2_univariate, '+-', err_deltam2_univariate)

