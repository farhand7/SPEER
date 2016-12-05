#!/usr/bin/env
__author__ = 'farhan_damani'

'''
	Likelihood computations for various models

'''
import logistic_regression as lr
import naive_bayes as nb
import numpy as np
import pandas as pd

#def _main():





def _RIVER_likelihood(e, g, beta, phi):
	# log p(z = 1 | g)
	log_p_z_1_given_g = lr.log_prob(g, beta) 
	# log p(z = 0 | g)
	log_p_z_0_given_g = np.log(1.0 - np.exp(log_p_z_1_given_g))
	# log p(e | z = 1)
	log_p_e_given_z_1 = nb.log_prob(e, 1, phi)
	# log p(e | z = 0)
	log_p_e_given_z_0 = nb.log_prob(e, 0, phi)
	import pdb; pdb.set_trace()
	#x_1 = 

	#m = np.maximum()



	return 1


def _multi_task_model_likelihood():
	return 1

def _logistic_regression_likelihood(posteriors):

	return np.sum(posteriors)


self.sigma_squared = 0.5


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