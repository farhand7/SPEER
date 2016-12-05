#!/usr/bin/env
__author__ = 'farhan_damani'

'''
	Compute posteriors for model benchmarks:
		1. SPEER without transfer
		2. RIVER
		3. Tissue specific genome only model
		4. Shared tissue genome only model

'''
import sklearn
import numpy as np
import pandas as pd
import cross_validation as cross_valid
import logistic_regression as lr
from sklearn import metrics
from sklearn import preprocessing


class BenchmarkPosteriors:

	def __init__(self, train_list, test_list, genomic_features):
		return 1






	'''
	def trainSharedGenomeOnlyModel(shared, annotations, expression, tissue_groups):
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
	'''

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





    def _concatenate_frames(self):

        # concat tissue data frames for global ROC curves

        train_all, test_all = pd.DataFrame(), pd.DataFrame()

        for i in range(self.num_tissues):
            train_all = train_all.append(self.train_list[i])
            test_all = test_all.append(self.test_list[i])

        return train_all, test_all



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