#!/usr/bin/env
__author__ = 'farhan_damani'

import numpy as np
import scipy.stats
import random
import os
import naive_bayes as nb
import logistic_regression as lr
import sys
import sklearn.linear_model
import sklearn
import copy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

THRESH = 1.5
class River:

    #def __init__(self, e_list, g_list, labels, e_list_test, g_list_test, labels_test):
    def __init__(self, train_list, test_list, genomic_features, output_dir=None):
    
        self.train_list = train_list
        self.test_list = test_list
        self.delta_likelihood = 10000000
        self.likelihood = 0
        #self.tissue_groups = tissue_groups

        ###########################################################
        # constants
        self.num_tissues = len(self.train_list)
        self.num_test_tissues = len(self.test_list)
        self.num_g_features = len(genomic_features)
        self.genomic_features = genomic_features
        self.model = 'RIVER'
        self.label = 'median_expr_label'
        
        ###########################################################
        # variables
        self.beta_parent = np.zeros(self.num_g_features) # 1 x M vector
        self.beta_children = np.zeros((self.num_tissues, self.num_g_features)) # K x M matrix

        self.phi = np.zeros((2,2))

        # hyperparameters
        self.lambda_hp_parent = 0.01
        self.lambda_hp_children = np.ones(self.num_tissues) # 1 x K
   
        #lambda_hp_children_dict = {'brain': 1,'group1': 1, 'group2': 1, 'group3': 1, 'group4': 3}
        lambda_hp_children_dict = {}
        lambda_hp_children_dict = {'shared': .01}

        for i in range(self.num_tissues):
            self.lambda_hp_children[i] = lambda_hp_children_dict['shared']
        
        # create directory for output
        if output_dir == None:
            self.directory = 'RIVER_output'
        else:
            self.directory = output_dir
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def run(self):
        '''
            Run EM model.

        '''        
        iter = 0
        self.initializeParameters() # initialize betas and phi from prior knowledge

        while True:
            #print('EM iter: ', iter)
            beta_children_old, beta_parent_old, phi_old = copy.copy(self.beta_children), copy.copy(self.beta_parent), copy.copy(self.phi)

            self.eStepGlobal() # E step

            self.mStep() # M - step
            # compute norm of new-old params
            beta_norm, phi_norm = self.computeBetaDiffNorm(beta_children_old, beta_parent_old), self.computePhiDiffNorm(phi_old)

            # check for convergence
            if (np.abs(beta_norm) < 1e-3 and np.abs(phi_norm) < 1e-3):                
                self.eStepGlobalTest() # compute p(z | test)

                self._write_to_file() # write data to file
                
                break
            iter+=1

        return self.train_list, self.test_list, self.beta_parent, self.beta_children, self.phi


    def _write_to_file(self):
        for i in range(self.num_tissues):
            self.train_list[i].to_csv(self.directory + '/train.csv')
            self.test_list[i].to_csv(self.directory + '/test.csv')

        np.savetxt(str(self.directory)+'/beta_children.txt', self.beta_children)
        np.savetxt(str(self.directory)+'/beta_parent.txt', self.beta_parent, newline=" ")
        np.savetxt(str(self.directory)+'/phi.txt', self.phi)

    def initializeParameters(self):
        '''
            Initialize betas from logistic regression and phi from prior knowledge

        '''
        # initialize betas from logistic regression
        for i in range(self.num_tissues):
            self.beta_children[i] = lr.sgd(self.train_list[i][self.genomic_features].values, self.train_list[i][self.label].values, self.getBetaLeaf(i), self.beta_parent, self.lambda_hp_children[i])
    
        
        self.phi = np.zeros((2,2))
        
        self.phi[0][0] = .8
        self.phi[1][0] = .2
        self.phi[0][1] = .3
        self.phi[1][1] = .7
        
    ################################################################################
    # helper functions

    def computeBetaDiffNorm(self, beta_children_old, beta_parent_old):
        '''
            Compute norm between betas at time step t-1 and t
        '''
        return np.linalg.norm(self.beta_children.ravel() - beta_children_old.ravel()) + np.linalg.norm(self.beta_parent - beta_parent_old)

    def computePhiDiffNorm(self, phi_old):
        '''
            Compute norm between phis at time step t-1 and t
        '''
        return np.linalg.norm(np.array(self.phi).ravel() - np.array(phi_old).ravel())


    def getBetaLeaf(self, i):
        return self.beta_children[i]

    ################################################################################
    # E-Step
    def eStepGlobal(self):
        '''
            E Step for each tissue
        '''
        for i in range(self.num_tissues):
            self.train_list[i][self.model] = self.eStepLocal(i, self.getBetaLeaf(i), self.phi)
        
    def eStepLocal(self, i, beta, phi):
        '''
           Compute expectation for tissue i
        '''
        # log P(Z = 1 | G)
        log_prob_z_1_given_g = lr.log_prob(self.train_list[i][self.genomic_features].values, beta)
        
        # log P(Z = 0 | G)
        log_prob_z_0_given_g = np.log(1.0 - np.exp(log_prob_z_1_given_g))
        
        # naive bayes
        log_prob_e_given_z_1 = nb.log_prob(self.train_list[i][self.label].values, 1, self.phi)        
        log_prob_e_given_z_0 = nb.log_prob(self.train_list[i][self.label].values, 0, self.phi)
        
        log_q = log_prob_e_given_z_1 + log_prob_z_1_given_g -  np.log(np.exp(log_prob_e_given_z_0) * np.exp(log_prob_z_0_given_g) + 
            np.exp(log_prob_e_given_z_1) * np.exp(log_prob_z_1_given_g))
        
        return np.exp(log_q)

    def eStepGlobalTest(self):
        for i in range(self.num_test_tissues):
            self.test_list[i][self.model] = self.eStepLocalTest(i, self.getBetaLeaf(i), self.phi)

    def eStepLocalTest(self, i, beta, phi):
        '''
           Compute expectation for tissue i
        '''
        # log P(Z = 1 | G)
        log_prob_z_1_given_g = lr.log_prob(self.test_list[i][self.genomic_features].values, beta)
        
        # log P(Z = 0 | G)
        log_prob_z_0_given_g = np.log(1.0 - np.exp(log_prob_z_1_given_g))

        # log P(E | Z = 1)
        log_prob_e_given_z_1 = nb.log_prob(self.test_list[i][self.label].values, 1, phi)
        # log P(E | Z = 0)
        log_prob_e_given_z_0 = nb.log_prob(self.test_list[i][self.label].values, 0, phi)
        log_q = log_prob_e_given_z_1 + log_prob_z_1_given_g -  np.log(np.exp(log_prob_e_given_z_0) * np.exp(log_prob_z_0_given_g) + np.exp(log_prob_e_given_z_1) * np.exp(log_prob_z_1_given_g))
        
        return np.exp(log_q)

    ######################################################################
    # M-Step
    def mStep(self):
        '''
            M Step for all k models
        '''
        # RIVER in parallel (e.g. l2 regularization on each tissue)
        self.estimateBeta()
        
        # phi update
        self.estimatePhis()
        
    # helper functions
    def estimateBeta(self):
        for i in range(self.num_tissues):
            self.beta_children[i] = lr.sgd(self.train_list[i][self.genomic_features].values, self.train_list[i][self.model].values, 
                self.getBetaLeaf(i), self.beta_parent, self.lambda_hp_children[i])
    
    def estimatePhis(self):
        '''
            Estimate phi counts in P(E|Z), using naive bayes
        '''

        x = 0
        for i in range(self.num_tissues): 
            x += nb.estimate_params(self.train_list[i][self.model].values, self.train_list[i][self.label].values)
        self.phi = x / float(self.num_tissues)
    

    def l1norm(self, a, b):
        '''
            Compute the l1 norm of a and b
            l1 norm = |a-b|
        '''
        return np.sum(np.abs(a-b))

    ######################################################################
'''
def main_simulation():
    annotations_path, expression_path, eqtl_path, ase_path, tissue_groups_path, z_labels_path = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
    train_list, test_list = [], []
    tissues = []

    annotations = pd.read_csv(annotations_path, header=None, index_col=None)


    expression = pd.read_csv(expression_path, index_col=(0))
    z_labels = pd.read_csv(z_labels_path, index_col=(0))
    assert(len(annotations) == len(expression))

    ase = pd.read_csv(ase_path, index_col=(0,1))
    eqtl = pd.read_csv(eqtl_path, index_col=(1,0))
    
    tissue_groups = processTissueGroups(tissue_groups_path)
    for k,v in tissue_groups.items():
        tissues.extend(v)
    annot_cols_original = list(annotations.columns)


    print ("processed all data...")

    #genomeonly_sharedtissue_beta = trainSharedGenomeOnlyModel(annotations, expression, tissue_groups)
    c = 0

    expression["median"] = -1
    # if at least 3 tissue groups are an outlier -> shared outlier
    outlier_indices = expression[expression.sum(axis=1) > 3].index
    expression.loc[outlier_indices, "median"] = 1
    normal_indices = expression[expression.sum(axis=1) <= 3].index
    expression.loc[normal_indices, "median"] = 0

    expr = expression["median"]
    expr.name = 'expression'

    z_labels["median"] = -1
    outlier_indices = z_labels[z_labels.sum(axis=1) > 3].index
    z_labels.loc[outlier_indices, "median"] = 1
    normal_indices = z_labels[z_labels.sum(axis=1) <= 3].index
    z_labels.loc[normal_indices, "median"] = 0

    #z = z_labels[group]
    z_med = z_labels["median"]
    z_med.name = 'z_labels'
    
    # concatenate annotations with expression data
    train = pd.concat([annotations, expr, z_med], axis=1).dropna()


    train["expr_label"] = expr
    # add posterior
    train["posterior"] = 0
    for group in tissue_groups:
        train["tissue"] = str(group)
    # random train/test split 80/20
    t = train.sample(frac=0.1, random_state = 200)
    test = train.drop(t.index)
    train = t


    train_list.append(train)
    
    test_list.append(test)


    ntwk = Network(train_list, test_list, tissue_groups, annot_cols_original)
    ntwk.run()
'''