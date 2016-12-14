#!/usr/bin/env
__author__ = 'farhan_damani'

'''
	Compute posteriors for model benchmarks:
		1. SPEER without transfer
		2. RIVER
		3. Tissue specific genome only model
		4. Shared tissue genome only model

'''
from sklearn import metrics
from sklearn import preprocessing
import cross_validation as cross_valid
import logistic_regression as lr
import numpy as np
import pandas as pd
import sklearn


class BenchmarkPosteriors:

	def __init__(self, train_list, test_list, genomic_features):

		self.train_list = train_list
		self.test_list = test_list
		self.genomic_features = genomic_features
		self.num_tissues = len(train_list)

	def fit_models(self):
		self.train_shared_tissue_genome_only()

		self.train_tissue_specific_genome_only()
		return self.train_list, self.test_list

	def train_shared_tissue_genome_only(self):

		beta = lr.sgd(self.train_list[0][self.genomic_features].values, self.train_list[0]["median_expr_label"].values, 
			np.zeros(len(self.genomic_features)), np.zeros(len(self.genomic_features)), 1.0)

		for i in range(self.num_tissues):
			self.train_list[i]["shared tissue genome only"] = np.exp(lr.log_prob(self.train_list[i][self.genomic_features].values, beta))
			self.test_list[i]["shared tissue genome only"] = np.exp(lr.log_prob(self.test_list[i][self.genomic_features].values, beta))

	def train_tissue_specific_genome_only(self):
		for i in range(self.num_tissues):
			# train tissue-specific model
			beta = lr.sgd(self.train_list[i][self.genomic_features].values, self.train_list[i]["expr_label"].values, 
				np.zeros(len(self.genomic_features)), np.zeros(len(self.genomic_features)), 1.0)
			self.train_list[i]["tissue specific genome only"] = np.exp(lr.log_prob(self.train_list[i][self.genomic_features].values, beta))
			self.test_list[i]["tissue specific genome only"] = np.exp(lr.log_prob(self.test_list[i][self.genomic_features].values, beta))


