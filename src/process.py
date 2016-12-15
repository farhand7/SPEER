#!/usr/bin/env
__author__ = 'farhan_damani'

'''
	Processes data and creates training matrices used as input to SPEER model.

	**confirm this works**

'''
import sklearn
import numpy as np
import pandas as pd
import cross_validation as cross_valid
import logistic_regression as lr
from sklearn import metrics
from sklearn import preprocessing
import sys

class Process:

	def __init__(self, input_dir, split_proportion=0.1):
		'''
			Process simulation data

			input_dir : str, default: None
				Input directory with expression, genomic annotations, and z labels

			split_proportion: float, default: 0.1
				Train/Test split proportion

		'''
		if input_dir == None:
			print("ERROR: Please enter valid input directory with the following files: e.csv, g.csv, and z.csv")
			sys.exit()
		self.input_dir = input_dir
		self.split_proportion = split_proportion

		# initialize train/test lists
		self.train_list = []
		self.test_list = []

		# genomic annotations - data already includes intercept term
		self.g = pd.read_csv(self.input_dir + 'g.csv', index_col=None, header=None)
		# expression data - columns indicate tissue groups
		self.e_labels = pd.read_csv(self.input_dir + 'e.csv', index_col=(0))
		# z labels
		self.z_labels = pd.read_csv(self.input_dir + 'z.csv', index_col=(0))
		# sanity check
		assert(len(self.g) == len(self.e_labels))
		assert(len(self.e_labels) == len(self.z_labels))

		self.tissues = list(self.e_labels.columns)
		self.genomic_features = list(self.g.columns)

	'''
	def __init__(self, annotations_path, expression_path, eqtl_path, tissue_groups_path, outlier_threshold=1.5):
		
			Process GTEx data into SPEER input

			add option to generate different types of train/test splits
				- random split
				- n1=n2 test data
				- no test data

			annotations_path : str, default = None
				Path to genomic annotations with following format:
					subject_id,gene_id,f1,f2,...,fN (N = # of annotations)
			...

			...


			outlier_threshold : float, default = 1.5
				Z-score threshold used to compute expression outlier status
		
		
		self.annotations_path = annotations_path
		self.expression_path = expression_path
		self.eqtl_path = eqtl_path
		self.tissue_groups_path = tissue_groups_path
		self.outlier_threshold = outlier_threshold

		self.train_list = []
		self.test_list = []
		self.tissues = []
		self.tissue_groups = {}


		self.g = pd.read_csv(self.annotations_path, index_col=(0,1))
		self.e = pd.read_csv(self.expression_path, index_col=(0,1))
		self.eqtl = pd.read_csv(self.eqtl_path, index_col=(1,0))

		self.tissue_groups = _process_tissue_groups(self.tissue_groups_path)
		for k,v in tissue_groups.items():
			self.tissues.extend(v)

		self.genomic_features = list(self.g.columns)
		self.genomic_features.insert(0, 'intercept')

		# scale genomic features
		self.g = self.g / (self.g.max() - self.g.min())
		annotation_columns = list(self.g.columns)


		for group in self.tissue_groups:
			expr_group = self.e[self.tissue_groups[group]]
			if group == 'brain':
				# require 50% tissues observed per sample (total 4 tissues)
				expr_group = expr_group.dropna(thresh = 3)
			elif group == 'group1':
				# require 50% tissues observed per sample (total 8 tissues)
				expr_group = expr_group.dropna(thresh = 4)
			elif group == 'shared':
				expr_group = expr_group.dropna(thresh = 5)
			else:
				expr_group = expr_group.dropna(thresh = 2)
			
			expr = np.abs(expr_group).median(axis=1)
			expr.name = 'expression'
			
			# concatenate annotations, expression, and eqtl data
			train = pd.concat([self.g, expr, self.eqtl], axis=1).dropna()

			train["expr_label"] = sklearn.preprocessing.binarize(np.abs(train["expression"].reshape(-1,1)), threshold = self.outlier_threshold)
			# add posterior column
			train["posterior"] = -1
			# add tissue column
			train["tissue"] = str(group)

			# generate train/test split
			train, test = self._generate_train_test(train, annotation_columns)
			# add intercept
			train.insert(0, 'intercept', 1)
			test.insert(0, 'intercept', 1)

			self.train_list.append(train)
			self.test_list.append(test)

			print ("processed ", group, " tissues.")
	'''
	def _process_simulated_data(self):
		# add shared tissue e and z labels
		self._compute_median()

		#for group in tissue_groups:
		for tissue in self.tissues:
			expr = self.e_labels[[tissue, "median"]]
			expr.columns = ['expr_label', "median_expr_label"]
			z = self.z_labels[[tissue, "median"]]
			z.columns = ['z_label', "median_z_label"]
			
			# concatenate annotations with expression data
			train = pd.concat([self.g, expr, z], axis = 1).dropna()
			#train["expr_label"] = expr[tissue]
			# add posterior columns
			train["SPEER"] = -1
			train["SPEER without transfer"] = -1
			train["RIVER"] = -1
			train["shared tissue genome only"] = -1
			train["tissue specific genome only"] = -1
			train["tissue"] = str(tissue)

			# random train/test split
			t = train.sample(frac = self.split_proportion, random_state = 200)
			test = train.drop(t.index)
			train = t

			self.train_list.append(train), self.test_list.append(test)

	def _compute_median(self):
		self.e_labels["median"] = -1
		# if at least 3 tissue groups are an outlier -> shared outlier
		outlier_indices = self.e_labels[self.e_labels.sum(axis=1) > 3].index

		self.e_labels.loc[outlier_indices, "median"] = 1
		normal_indices = self.e_labels[self.e_labels.sum(axis=1) <= 3].index
		self.e_labels.loc[normal_indices, "median"] = 0

		self.z_labels["median"] = -1
		outlier_indices = self.z_labels[self.z_labels.sum(axis=1) > 3].index
		self.z_labels.loc[outlier_indices, "median"] = 1
		normal_indices = self.z_labels[self.z_labels.sum(axis=1) <= 3].index
		self.z_labels.loc[normal_indices, "median"] = 0

	def _process_tissue_groups(self, path):
		tissue_groups = {}
		f = open(path)
		for l in f:
			w = l.strip().split(',')
			group = w[0]
			tissue_groups[group] = []
			for tissue in w[1:]: tissue_groups[group].append(tissue)
		return tissue_groups

	def _generate_train_test(self, train, annotation_columns):
		'''
			Training data contains annotation columns and other data columns
			annotation_columns is a list of genomic annotations
		'''
		annotation_columns.insert(0, 'gene_id')
		train.insert(0, 'gene_id', train.index.get_level_values('gene_id'))
		train.index = train.index.get_level_values('subject_id')

		# boolean mask - mark True for all duplicates and original
		duplicates_bool = train.duplicated(subset = annotation_columns, keep = False)
		# isolate training data w/ no duplicates - complement of boolean mask
		train_nodups = train[~duplicates_bool]
		train_nodups.index = [train_nodups.index, train_nodups['gene_id']]
		train_nodups = train_nodups.drop('gene_id', axis=1)

		# order duplicates consecutively
		duplicates = train[duplicates_bool].sort_values(by = annotation_columns)
		# remove odd duplicates
		duplicates = duplicates.groupby(by = annotation_columns).filter(lambda x: len(x) % 2 == 0)
		duplicates.index = [duplicates.index, duplicates['gene_id']]
		duplicates = duplicates.drop('gene_id', axis=1)
		n1 = duplicates.iloc[::2]
		n2 = duplicates.iloc[1::2]

		n1["gene_id"] = n1.index.get_level_values("gene_id")
		n1["N1_subject_id"] = n1.index.get_level_values("subject_id")
		n2["gene_id"] = n2.index.get_level_values("gene_id")
		n2["N2_subject_id"] = n2.index.get_level_values("subject_id")
		n2["N2_expr_label"] = n2["expr_label"]
		n1.index = [i for i in range(len(n1))]
		n2.index = [i for i in range(len(n2))]

		x = pd.concat([n1, n2[["N2_subject_id", "N2_expr_label"]]], axis=1)
		x.index = [x["N1_subject_id"], x["gene_id"]]
		x.index.names = ['subject_id', 'gene_id']

		return train_nodups, x
