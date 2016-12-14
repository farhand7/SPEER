#!/usr/bin/env
__author__ = 'farhan_damani'

'''
	K-fold cross validation to estimate 

'''
import sklearn
import numpy as np
import pandas as pd
import logistic_regression as lr
from sklearn import metrics
from sklearn import preprocessing

class Cross_Validation:

	def __init__(self, train_list, genomic_features, num_folds=None):
		'''
			:param num_folds (default=5) # of folds used to estimate lambdas
			:param train_list training data
			:genomic_features list of genomic features
		'''
		if num_folds == None:
			num_folds = 5
		self.num_folds = num_folds
		self.train_list = train_list
		self.num_tissues = len(train_list)
		# genomic features includes 'intercept'
		self.genomic_features = genomic_features

	def _cross_validate(self, G, E):
	    '''
	        K-fold Cross-validate beta MAP estimation to find optimal lambda
	        :param G genomic features
	        :param E expression labels
	        :lambda set 
	    '''
	    G = G
	    E = E
	    # lambda set
	    lambda_set = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
	    # initialize beta to zero
	    beta_init = np.zeros(len(self.genomic_features))
	    # AUC scores for each lambda and each fold
	    scores_list = np.zeros((len(lambda_set), self.num_folds))
	    # for each fold
	    for k in range(self.num_folds):
	    	# access training data (everything but Kth fold)
	        training = np.array([x for i, x in enumerate(G) if i % self.num_folds != k])
	        training_labels = np.array([x for i, x in enumerate(E) if i % self.num_folds != k])
	        # access validation data (Kth fold)
	        validation = np.array([[x for i, x in enumerate(G) if i % self.num_folds == k]])
	        validation_labels = np.array([x for i, x in enumerate(E) if i % self.num_folds == k])
	        # for each possible lambda
	        for i in range(len(lambda_set)):
	            # train a logistic regression model
	            beta = lr.sgd(training, training_labels, beta_init, beta_init, float(lambda_set[i]))
	            # compute predictions on validation set
	            scores = lr.log_prob(validation, beta).reshape(-1)
	            # compute auc using predictions and validation set labels
	            auc = sklearn.metrics.roc_auc_score(validation_labels, scores)
	            scores_list[i][k] = auc
	    # average across all folds for each lambda
	    lambda_averages = np.mean(scores_list, axis=1)
	    # sanity check
	    assert len(lambda_averages) == len(lambda_set)
	    optimal_lambda = lambda_set[np.argmax(lambda_averages)]
	    return optimal_lambda

	def _run_cross_validation(self):
		'''
			Cross-validate tissue-specific lambdas using beta MLE (with regularization)
			:param train_list list of training matrices indexed by tissue

		'''
		optimal_lambdas = [-1 for i in range(self.num_tissues)]
		# for each  tissue
		for j in range(self.num_tissues):
		    print("tissue: ", j)
		    train = self.train_list[j]
		    # access genomic annotations
		    g = train[self.genomic_features].values
		    # access expression labels
		    expr_label = train["expr_label"].values
		    # cross-validate lambdas
		    optimal_lambda = self._cross_validate(g, expr_label)
		    optimal_lambdas[j] = optimal_lambda
		
		return optimal_lambdas
