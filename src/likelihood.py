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


#if __name__ == "__main__":
#    main()