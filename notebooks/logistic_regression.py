from __future__ import print_function

import numpy as np
import sklearn.linear_model
from scipy.optimize.optimize import fmin_cg, fmin_bfgs, fmin
from scipy import linalg, optimize
from sklearn import cross_validation, metrics, linear_model
from datetime import datetime
import scipy as sp
import scipy.optimize
import matplotlib as mpl
import os

'''
	This script computes P(Z | G) using logistic regression and 
	completes a newton-raphson update.
'''
def log_prob(G, beta):
	'''
		log P(Z = 1 | G) = h(beta, x)
	'''
	a = np.log(sigmoid(beta, G))  # returns a 1 x N vector, where N = # of instances
	return a

def sigmoid(x, beta):
	return 1.0 / (1.0 + np.exp(-beta.dot(x.T)))

def _cost_function(vBeta, vBetaParent, mX, vY, lambda_hp):
	# cost function
	xr = (-(np.sum(vY*(np.dot(mX, vBeta) - np.log((1.0 + np.exp(np.dot(mX, vBeta))))) + (1-vY)*(-np.log((1.0 + np.exp(np.dot(mX, vBeta)))))))) + .5 * lambda_hp * (vBeta[1:] - vBetaParent[1:]).dot((vBeta[1:] - vBetaParent[1:]))
	# incur penalty from features excluding intercept
	#xr += .5 * lambda_hp * (vBeta[1:] - vBetaParent[1:]).dot((vBeta[1:] - vBetaParent[1:]))
	return xr

def _gradient_function(vBeta, vBetaParent, mX, vY, lambda_hp):
    # unregularized gradient objective
    a = (np.dot(mX.T, (sigmoid(mX, vBeta) - vY)))
    # incur penalty from features excluding intercept
    a[1:] += lambda_hp * (vBeta[1:] -vBetaParent[1:])
    return a

def sgd(G, Q, beta_c, beta_parent, lambda_hp):
	vBeta = beta_c
	vBetaParent = beta_parent
	mX = G
	vY = Q
	lambda_hp = lambda_hp
	beta_c = scipy.optimize.fmin_bfgs(_cost_function, x0 = vBeta, fprime = _gradient_function, args = (vBetaParent, mX, vY, lambda_hp), gtol = 1e-3, disp=True)	
	return beta_c
