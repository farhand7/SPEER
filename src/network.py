#!/usr/bin/env
__author__ = 'farhan_damani'

import numpy as np
import scipy.stats
import random
import os
import naive_bayes as nb
import logistic_regression as lr
#import logistic_regression_BACKUP as lr
import sys
import sklearn.linear_model
import sklearn
import copy
import pandas as pd
import timeit
import time
#import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys
import likelihood as ll


THRESH = 1.5
class Network:

    #def __init__(self, e_list, g_list, labels, e_list_test, g_list_test, labels_test):
    def __init__(self, train_list, test_list, tissue_groups, genomic_features):
    
        self.train_list = train_list
        self.test_list = test_list
        self.delta_likelihood = 10000000
        self.likelihood = 0
        self.tissue_groups = tissue_groups

        #self.genomeonly_sharedtissue_beta = genomeonly_sharedtissue_beta
        ###########################################################
        # constants
        self.num_tissues = len(self.train_list)
        self.num_test_tissues = len(self.test_list)
        self.num_g_features = len(genomic_features)
        self.genomic_features = genomic_features
        
        ###########################################################
        # variables
        self.beta_parent = np.zeros(self.num_g_features) # 1 x M vector
        self.beta_children = np.zeros((self.num_tissues, self.num_g_features)) # K x M matrix

        #beta_shared = pd.read_table('../input/shared_beta_20tissues.txt', delimiter=' ', header = None)
        #beta_shared = beta_shared.iloc[:,:-1].values.reshape(-1)
        #self.beta_parent = beta_shared


        self.phi = np.zeros((2,2))

        # hyperparameters
        #self.lambda_hp_parent = 0.01
        
        self.lambda_hp_parent = 4.333047702488766
        self.lambda_hp_children = np.ones(self.num_tissues) # 1 x K
        
        lambda_hp_children_dict = {}
        # simulation lambdas
        #lambda_hp_children_dict = {'brain': .01, 'group1': .01, 'muscle': .01, 'epithelial': .01, 'digestive': .01}
        #lambda_hp_children_dict = {'brain': 4, 'group1': 5, 'muscle': 6, 'epithelial': 7, 'digestive': 8}


        lambda_hp_children_dict = {'brain': 0.562229, 'group1': 0.75696656, 'muscle': 1.51665066, 'epithelial': 2.22486734, 'digestive': 5.17309994}

        for i in range(self.num_tissues):
            self.lambda_hp_children[i] = lambda_hp_children_dict[self.train_list[i].iloc[0]["tissue"]]

        self.sigma_squared = 0.5
        
        # create directory for output
        self.directory = str(sys.argv[1]) # first argument is output directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def run(self):
        '''
            Run EM model.

        '''        
        iter = 0
        # initialize betas and phi from prior knowledge
        self.initializeParameters()
        while True:
            print('EM iter: ', iter)
            beta_children_old, beta_parent_old, phi_old = copy.copy(self.beta_children), copy.copy(self.beta_parent), copy.copy(self.phi)
            # E-step
            self.eStepGlobal()
            # M-step
            self.mStep()
            # compute norm of new-old params
            beta_norm, phi_norm = self.computeBetaDiffNorm(beta_children_old, beta_parent_old), self.computePhiDiffNorm(phi_old)
            print('beta: ', beta_norm, 'phi: ', phi_norm)
            # check for convergence
            if (np.abs(beta_norm) < 1e-3 and np.abs(phi_norm) < 1e-3):              
                # compute expectation of test data
                    
                self.eStepGlobalTest()
                self._compute_baseline_posteriors()

                print('writing data to file...')
                train_all, test_all = self._concatenate_frames()
                import pdb; pdb.set_trace()
                # write concatenated frames to file
                train_all.to_csv(str(self.directory)+'/train.csv')
                test_all.to_csv(str(self.directory)+'/test.csv')
                
                #self.rocCurves(train_all, test_all)
                                
                print('writing data to file...')
                # save beta and phi parameters
                np.savetxt(str(self.directory)+'/beta_children.txt', self.beta_children)
                np.savetxt(str(self.directory)+'/beta_parent.txt', self.beta_parent, newline=" ")
                np.savetxt(str(self.directory)+'/phi.txt', self.phi)
                break
            iter+=1

    def hyperparameterEstimation(self):
        '''
            Hyperparameter estimation using bootstrap
            For 1 to K datasets:
                compute beta_i^c for each tissue
                compute beta_i^G by averaging all the above beta MLEs

            delta is a K x tissues x features matrix
            lambda is a tissues x features matrix


        '''
        return 1


    def _compute_baseline_posteriors(self):

        beta_river = pd.read_table('../input/beta_RIVER.v6.txt', header=None, delimiter=' ').values.reshape(-1)

        phi_river = np.loadtxt('../input/phi.RIVER.v7.csv')
        # read shared tissue genome only beta parameter
        #beta_genomeonly = pd.read_csv('genome_only_shared_beta.txt', header=None).values.reshape(-1)
        beta_shared = np.loadtxt('genome_only_shared_beta.v6.txt')
        import pdb; pdb.set_trace()
    
        # filter data to samples where there exists shared expression for RIVER  baseline
        f = open("../tissue_groups/shared.txt")
        x = f.readlines()
        tissues = x[0].strip().split(",")[1:]
        shared_expr = pd.read_csv(sys.argv[3], index_col=(0,1))
        # at least 2 out of all shared tissues need to be observed
        shared_expr = shared_expr[tissues].dropna(thresh = 2)
        shared_expr["median"] = np.abs(shared_expr).median(axis=1)
        shared_expr["shared_label"] = sklearn.preprocessing.binarize(np.abs(shared_expr["median"]).reshape(-1,1), threshold = THRESH)

        for i in range(self.num_test_tissues):
            self.train_list[i] = pd.concat([self.train_list[i], shared_expr["shared_label"]], axis=1).dropna()
            # limit to samples where posterior is not null
            #self.train_list[i] = self.train_list[i][self.train_list[i]["posterior"].notnull()]
            # limit to samples where there exists a shared expression label
            #self.train_list[i] = self.train_list[i].dropna()
            # concatenate shared labels with N=1 test data

            self.test_list[i] = pd.concat([self.test_list[i], shared_expr["shared_label"]], axis=1).dropna()
        
        # compute RIVER posteriors on tissue-specific cliques
        for i in range(self.num_test_tissues):
            # compute RIVER scores on training set
            self.train_list[i]['posterior_RIVER'] = self.eStepLocalRIVER(i, beta_river, phi_river)
            # compute RIVER scores on test set
            self.test_list[i]['posterior_RIVER'] = self.eStepLocalTestRIVER(i, beta_river, phi_river)
        #likelihood = ll._RIVER_likelihood(self.test_list[0][0]['expr_label'], self.test_list[0][0][self.genomic_features], beta_river, phi_river)
        

        # compute tissue-specific genome ony model posteriors on tissue-specific cliques
        for i in range(self.num_test_tissues):
            beta = lr.sgd(self.train_list[i][self.genomic_features].values, self.train_list[i]['expr_label'].values, np.zeros(self.num_g_features), np.zeros(self.num_g_features), 1)
            # compute predictions p(z|g) on training set
            self.train_list[i]['posterior_genome_only'] = np.exp(lr.log_prob(self.train_list[i][self.genomic_features].values, beta))
            # compute predictions p(z|g) on test set
            self.test_list[i]['posterior_genome_only'] = np.exp(lr.log_prob(self.test_list[i][self.genomic_features].values, beta))
        
        # compute shared tissue genome only model posteriors on tissue-specific cliques
        try:
            for i in range(self.num_test_tissues):
                self.train_list[i]['posterior_genome_only_shared'] = np.exp(lr.log_prob(self.train_list[i][self.genomic_features].values, beta_shared))
                self.test_list[i]['posterior_genome_only_shared'] = np.exp(lr.log_prob(self.test_list[i][self.genomic_features].values, beta_shared))
        except:
            import pdb; pdb.set_trace()

    def _compute_tissue_specific_ASE(self):
        #ase_path = sys.argv[5]
        #ase_df = pd.read_csv(ase_path, index_col=(0,1))

        return 1

    def _concatenate_frames(self):

        # concat tissue data frames for global ROC curves

        train_all, test_all = pd.DataFrame(), pd.DataFrame()

        for i in range(self.num_tissues):
            train_all = train_all.append(self.train_list[i])
            test_all = test_all.append(self.test_list[i])

        return train_all, test_all

    def rocCurves(self, train_all, test_all):

        
        print('writing data to file...')
        # write concatenated frames to file
        train_all.to_csv(str(self.directory)+'/train.csv')
        test_all.to_csv(str(self.directory)+'/n1.csv')
        test_all.to_csv(str(self.directory)+'/n2.csv')

        fpr, tpr, auc = dict(), dict(), dict()
        
        fpr[0], tpr[0], _ = sklearn.metrics.roc_curve(test_all["N2_expr_label"], test_all["posterior"])
        auc[0] = sklearn.metrics.auc(fpr[0], tpr[0])

        # ROC for RIVER
        fpr[1], tpr[1], _ = sklearn.metrics.roc_curve(test_all["N2_expr_label"], test_all["posterior_RIVER"])
        auc[1] = sklearn.metrics.auc(fpr[1], tpr[1])

        # ROC for tissue-specific genome only model
        fpr[2], tpr[2], _ = sklearn.metrics.roc_curve(test_all["N2_expr_label"], test_all["posterior_genome_only"])
        auc[2] = sklearn.metrics.auc(fpr[2], tpr[2])
        
        # ROC for shared tissue genome only model
        fpr[3], tpr[3], _ = sklearn.metrics.roc_curve(test_all["N2_expr_label"], test_all["posterior_genome_only_shared"])
        auc[3] = sklearn.metrics.auc(fpr[3], tpr[3])
        
        # ROC for N1 = N2 classifier
        fpr[4], tpr[4], _ = sklearn.metrics.roc_curve(test_all["N2_expr_label"], test_all["expr_label"])
        auc[4] = sklearn.metrics.auc(fpr[4], tpr[4])
        


        plt.figure()
        plt.style.use('seaborn-talk')
        plt.plot(fpr[0], tpr[0], label='Multi-task = {0:0.3f}'
                 ''.format(auc[0]), linewidth=2)
        plt.plot(fpr[1], tpr[1], label='RIVER = {0:0.3f}'
                 ''.format(auc[1]), linewidth=2)
        plt.plot(fpr[2], tpr[2], label='Tissue-spec. LR = {0:0.3f}'
                 ''.format(auc[2]), linewidth=2)
        plt.plot(fpr[3], tpr[3], label='Shared LR = {0:0.3f}'
                 ''.format(auc[3]), linewidth=2)
        plt.plot(fpr[4], tpr[4], label='N1=N2 = {0:0.3f}'
                 ''.format(auc[4]), linewidth=2)
        
        plt.plot([0,1], [0,1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(str(self.directory)+'/ROC.png')
        
   
        for group in self.tissue_groups:
            test = test_all[test_all["tissue"] == group]
            fpr, tpr, auc = dict(), dict(), dict()
            
            fpr[0], tpr[0], _ = sklearn.metrics.roc_curve(test["N2_expr_label"], test["posterior"])
            auc[0] = sklearn.metrics.auc(fpr[0], tpr[0])

            # ROC for RIVER
            fpr[1], tpr[1], _ = sklearn.metrics.roc_curve(test["N2_expr_label"], test["posterior_RIVER"])
            auc[1] = sklearn.metrics.auc(fpr[1], tpr[1])

            # ROC for tissue-specific genome only model
            fpr[2], tpr[2], _ = sklearn.metrics.roc_curve(test["N2_expr_label"], test["posterior_genome_only"])
            auc[2] = sklearn.metrics.auc(fpr[2], tpr[2])
            
            # ROC for shared tissue genome only model
            fpr[3], tpr[3], _ = sklearn.metrics.roc_curve(test["N2_expr_label"], test["posterior_genome_only_shared"])
            auc[3] = sklearn.metrics.auc(fpr[3], tpr[3])
            
            # ROC for N1 = N2 classifier
            fpr[4], tpr[4], _ = sklearn.metrics.roc_curve(test["N2_expr_label"], test["expr_label"])
            auc[4] = sklearn.metrics.auc(fpr[4], tpr[4])
            


            plt.figure()
            plt.style.use('seaborn-talk')
            plt.plot(fpr[0], tpr[0], label='Multi-task = {0:0.3f}'
                     ''.format(auc[0]), linewidth=2)
            plt.plot(fpr[1], tpr[1], label='RIVER = {0:0.3f}'
                     ''.format(auc[1]), linewidth=2)
            plt.plot(fpr[2], tpr[2], label='Tissue-spec. LR = {0:0.3f}'
                     ''.format(auc[2]), linewidth=2)
            plt.plot(fpr[3], tpr[3], label='Shared LR = {0:0.3f}'
                     ''.format(auc[3]), linewidth=2)
            plt.plot(fpr[4], tpr[4], label='N1=N2 = {0:0.3f}'
                     ''.format(auc[4]), linewidth=2)
            
            plt.plot([0,1], [0,1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            #plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.savefig(str(self.directory)+'/'+str(group)+'tissue_ROC.png')
            


        import pdb; pdb.set_trace()

    def identifyNonGeneticOutliers(self, threshold):
        for i in range(self.num_tissues):
            nonzero_indices = np.nonzero(self.train_list[i]['expr_label'])
            posteriors = [self.train_list[i]['posterior'][ind] for ind in nonzero_indices][0]
            print('posteriors less than .5 for expresion outliers: ', posteriors[posteriors < threshold])


    def initializeParameters(self):
        '''
            Initialize betas from logistic regression and phi from prior knowledge

        '''
        # initialize betas from logistic regression
        for i in range(self.num_tissues):
            self.beta_children[i] = lr.sgd(self.train_list[i][self.genomic_features].values, self.train_list[i]['expr_label'].values, self.getBetaLeaf(i), self.beta_parent, self.lambda_hp_children[i])
    
        self.phi = np.zeros(2)
        self.phi[0] = 0.7
        self.phi[1] = 0.6
        
        #self.phi = np.zeros((2,2))
        #self.phi = np.zeros((2,2,2))
        # indels and snvs are independent so we assume priors are similar for I=1 or I=0
        #self.phi[0] = .7
        #self.phi[1] = .6
        #self.phi[2] = .7
        
        '''
        self.phi[0][0][0] = .95
        self.phi[1][0][0] = .05
        self.phi[0][1][0] = .3
        self.phi[1][1][0] = .7

        self.phi[0][0][1] = .6
        self.phi[1][0][1] = .4
        self.phi[0][1][1] = .1
        self.phi[1][1][1] = .9
        '''

        '''
        
        self.phi[0][0] = .8
        self.phi[1][0] = .2
        self.phi[0][1] = .3
        self.phi[1][1] = .7
        
        '''
        
        '''
        # for simulation
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
            self.train_list[i]['posterior'] = self.eStepLocal(i, self.getBetaLeaf(i), self.phi)
    

    def eStepLocalRIVER(self, i, beta, phi):
        '''
           Compute expectation for tissue i
        '''
        # log P(Z = 1 | G)
        log_prob_z_1_given_g = lr.log_prob(self.train_list[i][self.genomic_features].values, beta)
        
        # log P(Z = 0 | G)
        log_prob_z_0_given_g = np.log(1.0 - np.exp(log_prob_z_1_given_g))
        
        # naive bayes
        log_prob_e_given_z_1 = nb.log_prob(self.train_list[i]['shared_label'].values, 1, phi)        
        log_prob_e_given_z_0 = nb.log_prob(self.train_list[i]['shared_label'].values, 0, phi)
        
        log_q = log_prob_e_given_z_1 + log_prob_z_1_given_g -  np.log(np.exp(log_prob_e_given_z_0) * np.exp(log_prob_z_0_given_g) + 
            np.exp(log_prob_e_given_z_1) * np.exp(log_prob_z_1_given_g))
        
        return np.exp(log_q)
    
    def eStepLocal(self, i, beta, phi):
        '''
           Compute expectation for tissue i
        '''
        # log P(Z = 1 | G)
        log_prob_z_1_given_g = lr.log_prob(self.train_list[i][self.genomic_features].values, beta)
        
        # log P(Z = 0 | G)
        log_prob_z_0_given_g = np.log(1.0 - np.exp(log_prob_z_1_given_g))
        
        # noisy OR
        log_prob_e_given_z_1 = nb.log_prob_noisyor_2_params(self.train_list[i]['expr_label'], 1, self.train_list[i]["eqtl"], phi)
        log_prob_e_given_z_0 = nb.log_prob_noisyor_2_params(self.train_list[i]['expr_label'], 0, self.train_list[i]["eqtl"], phi)

        
        # naive bayes
        #log_prob_e_given_z_1 = nb.log_prob(self.train_list[i]['expr_label'].values, 1, self.phi)        
        #log_prob_e_given_z_0 = nb.log_prob(self.train_list[i]['expr_label'].values, 0, self.phi)
        
        log_q = log_prob_e_given_z_1 + log_prob_z_1_given_g -  np.log(np.exp(log_prob_e_given_z_0) * np.exp(log_prob_z_0_given_g) + 
            np.exp(log_prob_e_given_z_1) * np.exp(log_prob_z_1_given_g))
        
        return np.exp(log_q)

    def eStepGlobalTest(self):
        for i in range(self.num_test_tissues):
            self.test_list[i]['posterior'] = self.eStepLocalTest(i, self.getBetaLeaf(i), self.phi)

    def eStepLocalTestRIVER(self, i, beta, phi):
        '''
           Compute expectation for tissue i
        '''
        # log P(Z = 1 | G)
        log_prob_z_1_given_g = lr.log_prob(self.test_list[i][self.genomic_features].values, beta)
        
        # log P(Z = 0 | G)
        log_prob_z_0_given_g = np.log(1.0 - np.exp(log_prob_z_1_given_g))
        
        # log P(E | Z = 1)
        log_prob_e_given_z_1 = nb.log_prob(self.test_list[i]['shared_label'].values, 1, phi)
        # log P(E | Z = 0)
        log_prob_e_given_z_0 = nb.log_prob(self.test_list[i]['shared_label'].values, 0, phi)
        log_q = log_prob_e_given_z_1 + log_prob_z_1_given_g -  np.log(np.exp(log_prob_e_given_z_0) * np.exp(log_prob_z_0_given_g) + np.exp(log_prob_e_given_z_1) * np.exp(log_prob_z_1_given_g))
        
        return np.exp(log_q) 

    def eStepLocalTest(self, i, beta, phi):
        '''
           Compute expectation for tissue i
        '''
        # log P(Z = 1 | G)
        log_prob_z_1_given_g = lr.log_prob(self.test_list[i][self.genomic_features].values, beta)
        
        # log P(Z = 0 | G)
        log_prob_z_0_given_g = np.log(1.0 - np.exp(log_prob_z_1_given_g))
        
        log_prob_e_given_z_1 = nb.log_prob_noisyor_2_params(self.test_list[i]['expr_label'], 1, self.test_list[i]["eqtl"], phi)
        log_prob_e_given_z_0 = nb.log_prob_noisyor_2_params(self.test_list[i]['expr_label'], 0, self.test_list[i]["eqtl"], phi)

        # log P(E | Z = 1)
        #log_prob_e_given_z_1 = nb.log_prob(self.test_list[i]['expr_label'].values, 1, phi)
        # log P(E | Z = 0)
        #log_prob_e_given_z_0 = nb.log_prob(self.test_list[i]['expr_label'].values, 0, phi)
        log_q = log_prob_e_given_z_1 + log_prob_z_1_given_g -  np.log(np.exp(log_prob_e_given_z_0) * np.exp(log_prob_z_0_given_g) + np.exp(log_prob_e_given_z_1) * np.exp(log_prob_z_1_given_g))
        
        return np.exp(log_q)

    ######################################################################
    # M-Step
    def mStep(self):
        '''
            M Step for all k models
        '''
        # beta update
        self.estimateBetas()
        
        # RIVER in parallel (e.g. l2 regularization on each tissue)
        #self.estimateBeta()
        
        # phi update
        self.estimatePhis()
        
    # helper functions
    def estimateBeta(self):
        for i in range(self.num_tissues):
            self.beta_children[i] = lr.sgd(self.train_list[i][self.genomic_features].values, self.train_list[i]['posterior'].values, 
                self.getBetaLeaf(i), self.beta_parent, self.lambda_hp_children[i])
    
    def estimateBetas(self):
        '''
            Beta estimation using coordinate gradient ascent
        '''
        iter = 0
        while True:

            beta_children_old, beta_parent_old = copy.copy(self.beta_children), copy.copy(self.beta_parent)

            # compute SGD for each beta child
            self.estimateBetaChildren()

            # update parent
            self.beta_parent = self.estimateBetaParent()
            #self.beta_parent = self.estimateBetaParentRIVERPrior()
            # compute norm of children + parent
            beta_norm = np.linalg.norm(self.beta_children.ravel() - 
                beta_children_old.ravel()) + np.linalg.norm(self.beta_parent - beta_parent_old)
            #print('local beta norm: ', beta_norm, iter)

            # convergence check
            if (beta_norm < 1e-3):
                break
            iter += 1

    def estimateBetaChildren(self):
        '''
            Helper function for estimateBetas()
            optimized using parallelization
        '''
        for i in range(self.num_tissues):
            self.beta_children[i] = lr.sgd(self.train_list[i][self.genomic_features].values, self.train_list[i]['posterior'].values, 
                self.getBetaLeaf(i), self.beta_parent, self.lambda_hp_children[i])

    def estimatePhis(self):
        '''
            Estimate phi counts in P(E|Z), using naive bayes
        '''

        x = 0
        for i in range(self.num_tissues): 
            #x += nb.estimate_params(self.train_list[i]['posterior'].values, self.train_list[i]['expr_label'].values)
            x += nb.estimate_params_noisyor_2_params(self.train_list[i]['expr_label'], self.train_list[i]['posterior'], self.train_list[i]['eqtl'])
        self.phi = x / float(self.num_tissues)
    
    def estimateBetaParent(self):
        '''
            Estimate beta parent 
            beta_j = (2 * \sum_c lambda^c * beta_j^c) / (2*lamda + L * \sum_c lambda^c)
        '''
        #import pdb; pdb.set_trace()        
        new =  (np.sum((np.array([self.lambda_hp_children]).T * self.beta_children), axis = 0)) / (self.lambda_hp_parent + np.sum(self.lambda_hp_children))
        '''
        old =  (np.sum((np.array([self.lambda_hp_children]).T * self.beta_children), axis = 0)) / ( 
            self.lambda_hp_parent + self.num_tissues * np.sum(self.lambda_hp_children))
        '''
        return new

    def estimateBetaParentRIVERPrior(self):
        '''
            Estimate beta parent 
            beta_j = (2 * \sum_c lambda^c * beta_j^c) / (2*lamda + L * \sum_c lambda^c)
        '''
        new =  (self.lambda_hp_parent * self.beta_river + np.sum((np.array([self.lambda_hp_children]).T * self.beta_children), axis = 0)) / (self.lambda_hp_parent + np.sum(self.lambda_hp_children))

        return new
    

    ######################################################################
    def log_p_beta(self):
        '''
            Compute P(Beta_j) = normal distribution, mean = 0, cov = sigma
            @param: j - jth component of beta
        '''
        log_prob = 0
        for j in xrange(1,self.num_g_features):
            log_prob += np.log(scipy.stats.norm(0, self.sigma_squared).pdf(self.beta_parent[j]))
        return log_prob

    def log_p_beta_child_given_beta(self, i):
        '''
            Compute P(Beta_child_j | Beta_j) = normal distribution, mean = beta_j, cov = sigma_child
            @param: i - tissue i
            @param: j - component
        '''
        log_prob = 0
        for j in xrange(1, self.num_g_features):
            log_prob += np.log(scipy.stats.norm(self.beta_parent[j],self.sigma_squared).pdf(self.beta_children[i][j]))
            #log_prob += np.log(scipy.stats.norm(0,self.sigma_squared).pdf(self.beta_children[i][j]))

        return log_prob

    ######################################################################
    def computeLikelihood(self):
        ll = self.log_p_beta()
        # P(beta^c | beta)
        for i in range(self.num_tissues):
            ll += self.log_p_beta_child_given_beta(i)
        for i in range(self.num_tissues):
            try:
                log_prob_z_1_g = lr.log_prob(self.train_list[i][self.genomic_features], self.getBetaLeaf(i)) 
                log_prob_z_0_g = np.log(1.0-np.exp(log_prob_z_1_g))
                
                log_prob_e_z_1 = nb.log_prob(self.train_list[i]['expr_label'], 1, self.phi)
                b = log_prob_e_z_1 + lr.log_prob(self.train_list[i][self.genomic_features], self.getBetaLeaf(i))
            except:
                continue
            a = nb.log_prob(self.train_list[i]['expr_label'], 0, self.phi) + np.log(1.0 - np.exp(log_prob_z_1_g))
            # log sum exp trick
            s = np.maximum(a, b)
            unnormalized_prob = s + np.log(np.exp(a - s) + np.exp(b - s))
            ll_tissue = np.nansum(unnormalized_prob)
            ll += ll_tissue

    def l1norm(self, a, b):
        '''
            Compute the l1 norm of a and b
            l1 norm = |a-b|
        '''
        return np.sum(np.abs(a-b))

    ######################################################################

def trainSharedGenomeOnlyModel(annotations, expression, tissue_groups):
    # concatenate all tissue groups together
    tissues = []
    for k,v in tissue_groups.items():
        tissues.extend(v)
    # limit expression data to tissues in groups and require at least 3 observed tissues per sample
    #import pdb; pdb.set_trace()
    #expression = expression[tissues].dropna(thresh = 5)
    #expression["median"] = np.abs(expression).median(axis=1)
    expression["expr_label"] = sklearn.preprocessing.binarize(np.abs(expression["median"]).reshape(-1,1), threshold = THRESH)
    train = pd.concat([annotations, expression["expr_label"]], axis = 1)
    train = train.dropna()
    train, test = generateTrainTest(train, list(annotations.columns))
    train.insert(0, 'intercept', 1)
    num_g_features = len(train.iloc[0,:-1])
    beta = lr.sgd(train.iloc[:,:-1], train.iloc[:,-1], np.ones(num_g_features), 
        np.zeros(num_g_features), 1.0)
    np.savetxt('genome_only_shared_beta.v6.txt', beta, newline=' ')
    return beta



def main_simulation():
    annotations_path, expression_path, eqtl_path, ase_path, tissue_groups_path, z_labels_path = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
    train_list, test_list = [], []
    tissues = []
    annotations = pd.read_csv(annotations_path, index_col=None, header=None)
    
    
    #annotations = annotations.drop("subject_id", axis=1)
    #annotations = annotations.drop("gene_id", axis=1)
    
    expression = pd.read_csv(expression_path, index_col=(0))
    z_labels = pd.read_csv(z_labels_path, index_col=(0))
    assert(len(annotations) == len(expression))

    ase = pd.read_csv(ase_path, index_col=(0,1))
    eqtl = pd.read_csv(eqtl_path, index_col=(1,0))
    
    tissue_groups = processTissueGroups(tissue_groups_path)
    for k,v in tissue_groups.items():
        tissues.extend(v)
    

    
    annot_cols_original = list(annotations.columns)
    #annot_cols_original.extend(list(annotations_tissue_specific.columns))
    
    #annot_cols_original.insert(0, 'intercept')
    


    # scale annotations and add intercept
    #annotations = annotations / (annotations.max() - annotations.min())
    #annotation_columns = list(annotations.columns)
    
    # limit expression samples to samples with corresponding annotations
    #expression = pd.concat([expression, annotations], axis=1)
    #expression = expression[expression[annotation_columns].notnull()]

    print ("processed all data...")

    #genomeonly_sharedtissue_beta = trainSharedGenomeOnlyModel(annotations, expression, tissue_groups)
    c = 0

    for group in tissue_groups:
        expr = expression[group]
        expr.name = 'expression'

        z = z_labels[group]
        z.name = 'z_labels'
        
        # concatenate annotations with expression data
        train = pd.concat([annotations, expr, z], axis=1).dropna()
        try:
            train["expr_label"] = expr
            # add posterior
            train["posterior"] = 0
            train["tissue"] = str(group)
            # random train/test split
        
            t = train.sample(frac=0.05, random_state = 200)
            test = train.drop(t.index)
            train = t
            # add intercept
            #train.insert(0, 'intercept', 1)
            
            #test.insert(0, 'intercept', 1)

            train_list.append(train)
            
            test_list.append(test)

            print ("processed ", group, " tissues.")
        except:
            import pdb; pdb.set_trace()

        c+=1
    import pdb; pdb.set_trace()
    ntwk = Network(train_list, test_list, tissue_groups, annot_cols_original)
    ntwk.run()


def main():

    annotations_path, expression_path, eqtl_path, ase_path, tissue_groups_path = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
    train_list, test_list = [], []
    tissues = []
    annotations = pd.read_csv(annotations_path, index_col=(0,1))
    expression = pd.read_csv(expression_path, index_col=(0,1))
    ase = pd.read_csv(ase_path, index_col=(0,1))
    eqtl = pd.read_csv(eqtl_path, index_col=(1,0))
    #annotations_tissue_specific = pd.read_csv(annotations_tissue_spec_path, index_col=(0,1))

    tissue_groups = processTissueGroups(tissue_groups_path)
    for k,v in tissue_groups.items():
        tissues.extend(v)
    annot_cols_original = list(annotations.columns)
    #annot_cols_original.extend(list(annotations_tissue_specific.columns))
    annot_cols_original.insert(0, 'intercept')
    # scale annotations and add intercept
    annotations = annotations / (annotations.max() - annotations.min())
    annotation_columns = list(annotations.columns)

    # limit expression samples to samples with corresponding annotations
    #expression = pd.concat([expression, annotations], axis=1)
    #expression = expression[expression[annotation_columns].notnull()]

    print ("processed all data...")

    genomeonly_sharedtissue_beta = trainSharedGenomeOnlyModel(annotations, expression, tissue_groups)
    c = 0

    for group in tissue_groups:
        expr_group = expression[tissue_groups[group]]
        if group == 'brain':
            expr_group = expr_group.dropna(thresh = 3)
        elif group == 'group1':
            expr_group = expr_group.dropna(thresh = 4)
        elif len(tissue_groups[group]) == 1:
            expr_group = expr_group.dropna()
        else:
            expr_group = expr_group.dropna(thresh = 2)
        
        expr = np.abs(expr_group).median(axis=1)
        expr.name = 'expression'
        
        train = pd.concat([annotations, expr, eqtl], axis=1).dropna()
        # concatenate annotations with expression data
        #train = pd.concat([annotations, expr], axis=1).dropna()

        try:
            '''
            expr_group = expr_group.dropna()
            # compute med(abs(z-score)) for each sample
            expr_group["expr_median"] = np.abs(expr_group)
            
            # concatenate generic annotations, tissue-specific annotatoins, and median of expression
            #train = pd.concat([annotations, annotations_tissue_specific, expr_group["expr_median"]], axis = 1)
            train = pd.concat([annotations, expr_group["expr_median"]], axis = 1)
            # drop samples with any missing annotations
            train = train.dropna()
            
            train = pd.concat([train, eqtl], axis=1)
            train = train.dropna()

            # fill missing eqtl data with 0
            #train = train.fillna(value = 0)
            # add binarized expression label
            '''
            train["expr_label"] = sklearn.preprocessing.binarize(np.abs(train["expression"].reshape(-1,1)), threshold = THRESH)
            # add posterior
            train["posterior"] = 0
            train["tissue"] = str(group)

            train, test = generateTrainTest(train, annotation_columns)
            # add intercept
            train.insert(0, 'intercept', 1)
            
            test.insert(0, 'intercept', 1)

            train_list.append(train)
            
            test_list.append(test)

            print ("processed ", group, " tissues.")
        except:
            import pdb; pdb.set_trace()
        c+=1
    ntwk = Network(train_list, test_list, tissue_groups, annot_cols_original)
    ntwk.run()


##### main helper functions #####
def processTissueGroups(tissue_groups_path):
    tissue_groups = {}
    f = open(tissue_groups_path)
    for l in f:
        w = l.strip().split(',')
        group = w[0]
        tissue_groups[group] = []
        for tissue in w[1:]: tissue_groups[group].append(tissue)
    return tissue_groups

def generateTrainTest(train, annotation_columns):
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

if __name__ == "__main__":
    #main_simulation()
    main()