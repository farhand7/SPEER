#!/usr/bin/env
__author__ = 'farhan_damani'

import copy
import logistic_regression as lr
import naive_bayes as nb
import numpy as np
import os
import pandas as pd
import sklearn
import sys

class Network:

    def __init__(self, train_list, test_list, tissue_groups, genomic_features, with_transfer=True, output_dir = None, lambda_hp_parent=None, 
        lambda_hp_children_dict=None, e_distribution=None):
        '''
            train_list, test_list : list
                List of training matrices containing genomic features, median_expression, expression_outlier_label, eqtl (optional), posterior, tissue

            tissue_groups : dictionary of lists
                Keys - tissue groups
                Values - list of tissues for each group

            genomic_features : list
                list of genomic features (starts with 'intercept')

            with_transfer : boolean, default : True
                Train SPEER (True) or SPEER without parameter transfer (False)

            output_dir : str, default : None
                Output directory

            lambda_hp_parent : float, default : None
                Lambda transfer factor. If parent transfer factor is not specified, we use 4.33304770248876, which is estimate used to produce GTEx results.

            lambda_hp_children_dict : dictionary of floats, default: None
                tissue-specific transfer factors
                Keys - tissue groups
                Values - transfer factors
                If children transfer factors are not specified, we use {'brain': 0.562229, 'group1': 0.75696656, 'muscle': 1.51665066, 
                'epithelial': 2.22486734, 'digestive': 5.17309994}. See paper for details.

            e_distribution : str, {'cat', 'noisyor'}, default: None
                Distribution on e can be either 'cat' or 'noisyor'. If the option is 'cat', then we use a categorical distribution
                If the option is 'noisyor', we use a Noisy Or distribution. If neither are specified, we use a NoisyOr if e has at least
                two parent variables (e.g. r and eqtl). If e has only one parent variable, we use a categorical distribution.

        '''

        self.train_list = train_list
        self.test_list = test_list
        self.tissue_groups = tissue_groups
        self.genomic_features = genomic_features
        
        if lambda_hp_parent == None:
            #self.lambda_hp_parent = 4.333047702488766
            self.lambda_hp_parent = 0.01
        else:
            self.lambda_hp_parent = lambda_hp_parent
        
        if lambda_hp_children_dict == None:
            #self.lambda_hp_children_dict = {'Brain': 0.562229, 'group1': 0.75696656, 'Muscle': 1.51665066, 'Epithelial': 2.22486734, 'Digestive': 5.17309994}
            #self.lambda_hp_children_dict = {'brain': 0.562229, 'group1': 0.75696656, 'muscle': 1.51665066, 'epithelial': 2.22486734, 'digestive': 5.17309994}
            #self.lambda_hp_children_dict = {'brain': 0.01, 'group1': 0.01, 'muscle': 0.01, 'epithelial': 0.01, 'digestive': 0.01}
            self.lambda_hp_children_dict = {'brain': 4, 'group1': 5, 'muscle': 6, 'epithelial': 7, 'digestive': 8}



        else:
            self.lambda_hp_children_dict = lambda_hp_children_dict

        # SPEER vs SPEER w/o transfer (True or False)
        self.with_transfer = with_transfer

        if self.with_transfer:
            self.model = 'SPEER'
        else:
            self.model = 'SPEER without transfer'

        if e_distribution == None:
            # check if training data contains eqtl column
            if 'eqtl' in self.train_list[0].columns:
                self.e_distribution = 'noisyor'
                #self.phi = np.zeros(2)
            else:
                self.e_distribution = 'cat'
                #self.phi = np.zeros((2,2))
        else:
            self.e_distribution = e_distribution
        
        if self.e_distribution == 'noisyor':
            self.phi = np.zeros(2)
        else:
            self.phi = np.zeros((2,2))


        self.num_tissues = len(self.train_list)
        self.num_test_tissues = len(self.test_list)
        self.genomic_features = genomic_features
        self.num_g_features = len(genomic_features)

        
        self.beta_parent = np.zeros(self.num_g_features) # 1 x M vector
        self.beta_children = np.zeros((self.num_tissues, self.num_g_features)) # K x M matrix

        self.lambda_hp_children = []
        # based on ordering of processed tissues, create list of tissue-specific transfer factors
        for i in range(self.num_tissues):
            #self.lambda_hp_children[i] = self.lambda_hp_children_dict[self.train_list[i].iloc[0]["tissue"]]
            self.lambda_hp_children.append(self.lambda_hp_children_dict[self.train_list[i].iloc[0]["tissue"]])
        
        # create directory for output
        if output_dir == None:
            self.directory = 'SPEER_output'
        else:
            self.directory = output_dir
        # check if directory already exists
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def run(self):
        '''
            Optimize model parameters using EM algorithm.

        '''        
        iter = 0
        # initialize betas and phi from prior knowledge
        self.initializeParameters()
        while True:
            #print('EM iter: ', iter)
            beta_children_old, beta_parent_old, phi_old = copy.copy(self.beta_children), copy.copy(self.beta_parent), copy.copy(self.phi)
            # E-step
            self.eStepGlobal()
            # M-step
            self.mStep()
            # compute norm of new-old params
            beta_norm, phi_norm = self.computeBetaDiffNorm(beta_children_old, beta_parent_old), self.computePhiDiffNorm(phi_old)
            #print('beta: ', beta_norm, 'phi: ', phi_norm)
            
            # convergence check
            if (np.abs(beta_norm) < 1e-3 and np.abs(phi_norm) < 1e-3):              
                # compute expectation of test data
                self.eStepGlobalTest()

                # write train and test data to file
                self._write_to_file()
                break
            iter+=1

        return self.train_list, self.test_list, self.beta_parent, self.beta_children, self.phi
    
    def _write_to_file(self):

        for i in range(self.num_tissues):
            tissue = self.train_list[0]["tissue"].iloc[0]
            self.train_list[i].to_csv(self.directory + '/train_' + tissue + '.csv')
            self.test_list[i].to_csv(self.directory + '/test_' + tissue + '.csv')

        np.savetxt(self.directory + '/beta_children.txt', self.beta_children)
        np.savetxt(self.directory + '/beta_parent.txt', self.beta_parent, newline=" ")
        np.savetxt(self.directory + '/phi.txt', self.phi)

    def initializeParameters(self):
        '''
            Initialize betas from logistic regression and phi from prior knowledge

        '''
        # initialize betas from logistic regression where tissue-specific expression outlier status is label
        for i in range(self.num_tissues):
            self.beta_children[i] = lr.sgd(self.train_list[i][self.genomic_features].values, self.train_list[i]['expr_label'].values, self.getBetaLeaf(i), 
                self.beta_parent, self.lambda_hp_children[i])
        
        if self.e_distribution == 'noisyor':
            # p(e = 1 | z = 1)
            self.phi[0] = 0.7
            # p(e = 1 | eqtl = 1)
            self.phi[1] = 0.6
        else:
            # p(e = 0 | z = 0)
            self.phi[0][0] = .8
            # p(e = 1 | z = 0)
            self.phi[1][0] = .2
            # p(e = 0 | z = 1)
            self.phi[0][1] = .3
            # p(e = 1 | z = 1)
            self.phi[1][1] = .7
     
        
        # for simulation
        '''
        self.phi[0][0] = .65
        self.phi[1][0] = .35
        self.phi[0][1] = .4
        self.phi[1][1] = .6
        '''


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
            self.train_list[i][self.model] = self.eStepLocal(i, self.train_list[i], self.getBetaLeaf(i), self.phi)

    def eStepGlobalTest(self):
        '''
            P(z | test data)
        '''
        for i in range(self.num_test_tissues):
            self.test_list[i][self.model] = self.eStepLocal(i, self.test_list[i], self.getBetaLeaf(i), self.phi)

    def eStepLocal(self, i, data, beta, phi):
        '''
           Compute p(z | ...) for tissue i
        '''
        # log p(z | g)
        log_prob_z_1_given_g = lr.log_prob(data[self.genomic_features].values, beta)        
        log_prob_z_0_given_g = np.log(1.0 - np.exp(log_prob_z_1_given_g))
        
        # log p(e | z, q)
        if self.e_distribution == 'noisyor':
            # noisy OR
            log_prob_e_given_z_1 = nb.log_prob_noisyor_2_params(data['expr_label'], 1, data["eqtl"], phi)
            log_prob_e_given_z_0 = nb.log_prob_noisyor_2_params(data[i]['expr_label'], 0, data["eqtl"], phi)
        # log p(e | z)
        else:
            # naive bayes
            log_prob_e_given_z_1 = nb.log_prob(data['expr_label'].values, 1, self.phi)        
            log_prob_e_given_z_0 = nb.log_prob(data['expr_label'].values, 0, self.phi)
           
        # p(e|z =1) * p(z = 1 | g) / (\sum_{z \in S} p(z = s | g) * p(e | z = s))
        log_q = log_prob_e_given_z_1 + log_prob_z_1_given_g -  np.log(np.exp(log_prob_e_given_z_0) * np.exp(log_prob_z_0_given_g) + 
            np.exp(log_prob_e_given_z_1) * np.exp(log_prob_z_1_given_g))
        
        return np.exp(log_q)

    ######################################################################################################################################
    # M-Step
    def mStep(self):
        '''
            Maximization step.
        '''
        # beta MAP estimation
        self.estimateBetas()

        # phi MAP estimation
        self.estimatePhis()
      

    def estimateBetas(self):
        # blocked coordinate gradient descent if estimating dependent parameters
        if self.with_transfer:
            self._blocked_coordinate_gradient_descent()
        # estimate tissue-specific betas independently using gradient descent
        else:
            self._gradient_descent()

    def _gradient_descent(self):
        for i in range(self.num_tissues):
            self.beta_children[i] = lr.sgd(self.train_list[i][self.genomic_features].values, self.train_list[i][self.model].values, 
                self.getBetaLeaf(i), self.beta_parent, self.lambda_hp_children[i])
   
    def _blocked_coordinate_gradient_descent(self):
        '''
            Beta estimation using coordinate gradient ascent
        '''
        while True:

            beta_children_old, beta_parent_old = copy.copy(self.beta_children), copy.copy(self.beta_parent)

            # estimate each child beta - p(z|g; beta)
            self._gradient_descent()

            # update parent
            self.beta_parent = self._estimate_beta_parent()

            # compute norm of children + parent
            beta_norm = np.linalg.norm(self.beta_children.ravel() - 
                beta_children_old.ravel()) + np.linalg.norm(self.beta_parent - beta_parent_old)

            # convergence check
            if (beta_norm < 1e-3):
                break

    def _estimate_beta_parent(self):
        '''
            Estimate beta parent 
            beta_j = (2 * \sum_c lambda^c * beta_j^c) / (2*lamda + L * \sum_c lambda^c)
        '''
        return (np.sum((np.array([self.lambda_hp_children]).T * self.beta_children), axis = 0)) / (self.lambda_hp_parent + np.sum(self.lambda_hp_children))    


    def estimatePhis(self):
        '''
            Phi estimation.
        '''

        x = 0
        for i in range(self.num_tissues):
            if self.e_distribution == 'noisyor':
                x += nb.estimate_params_noisyor_2_params(self.train_list[i]['expr_label'], self.train_list[i][self.model], self.train_list[i]['eqtl'])
            else:
                x += nb.estimate_params(self.train_list[i][self.model].values, self.train_list[i]['expr_label'].values)
        self.phi = x / float(self.num_tissues)
    
    def l1norm(self, a, b):
        '''
            Compute the l1 norm of a and b
            l1 norm = |a-b|
        '''
        return np.sum(np.abs(a-b))

    ######################################################################################################################################