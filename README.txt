#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 23:23:53 2022

@author: claraaldegundemanteca
"""

Project 1.py: In cludes the data and simulation hitograms, as well as the 
NLL and probability functions. It gives an overview of the dataset and how the
NLL behaves. Throughout, alpha is treated as the rate of change of neutrino
iteraction cross section, and the code is written in a way such that
NLL_alpha function could be generally used throughout. It
takes alpha  = 1 unless stated (energy scaling is not taken
into account unless specifically indicated).

Univariate.py: Computes 1D parabolic minimisation for both deltam2 and theta23 
and combines these functions into a single univariate algorithm. The result is 
plotted on a contour plot for a better understanding.

It also includes the functions to calculata sigma with respect to the 3 parameters
we're interested in (theta23, deltam2 and alpha), (defined as 
a change of 0.5 in the NLL assuming a parabolic shape).

Gradient descent.py: Performs the gradient descent minimisation algorithm, 
including all the partial derivatives needed.

Simulated annealing.py: Includes the simulated annealing method, as well as
convergence tests and relevant plots (e. g. results after each cycle plotted
onto NLL contours)

Annealing test.py: Applies the annealing algorithm to a known function 
(we used cosh x + cosh y) to check it behaves as expected


Uniariate test.py: Applies the univariate algorithm to a known function 
(we used cosh x + cosh y) to check it behaves as expected