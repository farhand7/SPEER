#!/usr/bin/env
__author__ = 'farhan_damani'

'''
	Bootstrap estimation procedure to estimate tissue-specific transfer factors and global transfer factor

	**confirm this works**

'''
import sklearn
import numpy as np
import pandas as pd
import cross_validation as cross_valid
import logistic_regression as lr
from sklearn import metrics
from sklearn import preprocessing

class Bootstrap:

	def __init__(self, train_list, tissues, genomic_features, num_simulations=None, num_folds=None):
		'''
			:param num_simulations number of bootstrap simulations
			:param num_folds number of folds used in K-fold cross-validation to estimate lambdas used in each bootstrap simulation
			:param train_list training data 
		'''
		if num_simulations == None:
			num_simulations = 100
		if num_folds == None:
			num_folds = 5
		self.num_simulations = num_simulations
		self.num_folds = num_folds
		self.train_list = train_list
		self.num_tissues = len(train_list)
		self.tissues = tissues
		# genomic features includes 'intercept'
		self.genomic_features = genomic_features
		self.num_features = len(genomic_features)
		
		# simulate each dataset 
		bootstrap_data = [self.bootstrap_resample(X) for X in self.train_list]
		# cross-validate using bootstrapped data
		cv = cross_valid.Cross_Validation(bootstrap_data, self.genomic_features, self.num_folds)
		self.optimal_lambdas = cv._run_cross_validation()
		#self.optimal_lambdas = [1,1,1,1,1]

	def bootstrap_resample(self, X, n=None):
		""" 
		citation: http://nbviewer.jupyter.org/gist/aflaxman/6871948
		Bootstrap resample an array_like
		Parameters
		----------
		X : array_like
		  data to resample
		n : int, optional
		  length of resampled array, equal to len(X) if n==None
		Results
		-------
		returns X_resamples
		"""
		if n == None:
			n = len(X)
			
		resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
		X_resample = X.iloc[resample_i]
		return X_resample


	def estimateBetaParent(self, beta_children, lambda_hp_children, lambda_hp_parent):
		'''
			Estimate beta parent 
			beta_j = (2 * \sum_c lambda^c * beta_j^c) / (2*lamda + L * \sum_c lambda^c)
		'''
		return (np.sum((np.array([lambda_hp_children]).T * beta_children), axis = 0)) / (lambda_hp_parent + np.sum(lambda_hp_children))


	def computeEmpiricalVariance(self, delta, K):
		lambda_hp = np.zeros((self.num_tissues, self.num_features - 1))
		# for each tissue
		for t in range(self.num_tissues):
			# for each feature (excluding intercept)
			for j in range(self.num_features - 1):
				lambda_hp[t][j] = np.sum(delta[:,t,j]**2) / (K-1)
		return lambda_hp

	def computeEmpiricalVarianceParent(self, delta, K):
		lambda_hp = np.zeros(self.num_features - 1)
		for j in range(self.num_features - 1):
			lambda_hp[j] = np.sum(delta[:,j]**2) / (K-1)
		return lambda_hp


	def _run_bootstrap(self):

		# beta is a T x M matrix, where T = # of tissues and M = number of features (not including intercept)
		beta = np.zeros((self.num_simulations, self.num_tissues, self.num_features - 1))
		beta_parent = np.zeros(self.num_features - 1)
		beta_init = np.zeros(self.num_features)

		delta = np.zeros((self.num_simulations, self.num_tissues, self.num_features - 1))
		delta_parent = np.zeros((self.num_simulations, self.num_features - 1))
		lambda_set = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])

		# for each tissue
		for j in range(self.num_tissues):
			# generate K random data sets
			optimal_lambda = self.optimal_lambdas[j]
			# for each simulation
			for i in range(self.num_simulations):
				# generate simulated dataset i for tissue j
				train_sample = self.bootstrap_resample(self.train_list[j])
				g = train_sample[self.genomic_features]
				expr_label = train_sample["expr_label"]
				#optimal_lambda = _cross_validate(g, expr_label, np.zeros(len(annot_cols_original)), np.zeros(len(annot_cols_original)), lambda_set)
				# compute L2 regularized logistic regression and store non-intercept terms
				beta[i][j] = lr.sgd(g.values, expr_label.values, beta_init, beta_init, optimal_lambda)[1:]
		# for each simulation
		for i in range(self.num_simulations):
			# estimate beta parent as an equally weighted average of its children
			beta_parent = self.estimateBetaParent(beta[i], np.ones(self.num_tissues), 1)
			# estimate difference between children betas and parent beta
			for j in range(self.num_tissues):
				delta[i][j] = (beta[i][j] - beta_parent)
			delta_parent[i] = beta_parent
		# estimate empirical variance between computed differences of children betas and parent beta
		lambda_inverse = self.computeEmpiricalVariance(delta, i+1)
		# compute the average of the feature-specific variances
		lambda_inverse = np.sum(lambda_inverse, axis=1) / lambda_inverse.shape[1]

		# compute empirical variance between computed differences of parent beta and zero vector
		lambda_parent_inverse = self.computeEmpiricalVarianceParent(delta_parent, i+1)
		# compute average of feature-specific variances
		lambda_parent_inverse = np.sum(lambda_parent_inverse) / lambda_parent_inverse.shape[0]
				
		lambda_hp_children = 1.0 / lambda_inverse
		lambda_hp_parent = 1.0 / lambda_parent_inverse
		
		# mapping from tissue to estimated transfer factor
		lambda_hp_children_dict = {}
		for i,tissue in enumerate(self.tissues):
			lambda_hp_children_dict[tissue] = lambda_hp_children[i]

		return lambda_hp_children_dict, lambda_hp_parent

