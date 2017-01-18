#!/usr/bin/env
__author__ = 'farhan_damani'
'''
	Simulate data from SPEER

'''
import pandas as pd
import numpy as np
import logistic_regression as lr
import matplotlib.pyplot as plt
import os

class SimulateData:

	def __init__(self, output_dir, model_type, mu_z0, mu_z1, lambda_parent=None):
		'''
			:param output_dir: directory to output simulated data
			:param model_type: 'with_transfer' or 'without_transfer'
			:param lambda_parent: 

		'''


		self.output_dir = output_dir
		self.model_type = model_type
		self.lambda_parent = lambda_parent
		self.mu_z0 = mu_z0
		self.mu_z1 = mu_z1
		self.tissues = ['brain', 'group1', 'muscle', 'epithelial', 'digestive']
		self.num_features = 21 # including intercept


		# data
		self.e = None
		self.g = None
		self.z = None


	def _simulate_g(self):
		'''
			g ~ uniform(0,1)
			simulate 1000 samples with 20 features
		'''
		g = np.random.uniform(0,1, size=(10000,20))
		intercept = np.ones((10000,1))
		return np.concatenate((intercept, g), axis=1)

	def _simulate_alpha(self, sd):
		'''
			Generate alpha parameter where alpha ~ N(0, sigma^2)
		'''
		return np.random.normal(np.zeros(self.num_features), sd)

	def _simulate_beta_given_alpha(self, alpha, sd):
		'''
			Generate beta | alpha ~ N(alpha, sigma^2)
		'''
		return np.random.normal(alpha, sd)

	def _simulate_beta_no_parent(self, sd):
		'''
			Generate beta ~ N(0, sigma^2)
		'''
		return np.random.normal(np.zeros(self.num_features), sd)

	def _simulate_z_given_g(self):
		'''
			z | g ~ Bin(psi(beta,g))
			psi ~ 1/(1+e^{-beta.T.G})
		'''
		p_z_given_g = self._compute_p_z_given_g(beta, g)
		return np.random.binomial(1, p_z_given_g)

	def _compute_p_z_given_g(self, beta, g):
		'''
			P(z | g; beta)
		'''
		return np.exp(lr.log_prob(g, beta))

	def _simulate_e_given_z(self, z, phi):
		'''
			e | z ~ Bernoulli(phi(z))
		'''
		return 1

	def _simulate_phi(self, mu_z0, mu_z1, var_z0, var_z1):
		'''
			Using a beta distribution, sample phi counts
			
			fix this function
			
		'''
		# p(e=1|z=0), p(e=1|z=1)
		phi = np.zeros(2)
		
		alpha, beta = self._compute_beta_hyperparameters(mu_z0, var_z0)
		
		# sample from Beta
		phi[0] = np.random.beta(alpha, beta)
		
		alpha, beta = self._compute_beta_hyperparameters(mu_z1, var_z1)

		# sample from Beta
		phi[1] = np.random.beta(alpha, beta)
		
		return phi

	def _compute_beta_hyperparameters(self, mu, var):
		'''
			Given mean and variance, compute parameters to Beta distribution: alpha, beta
		'''
		alpha = (((1 - mu) / var) - (1 / mu)) * mu**2
		beta = alpha * (1 / mu - 1)
		return alpha, beta

	def _simulate_RIVER(self):
		lambda_hp = 4.333047702488766
		alpha_variance = 1.0 / lambda_hp_parent
		# simulate beta
		beta = self._simulate_beta_no_parent(alpha_variance)
		# compute p(z | g, beta)
		p_z_given_g = self._compute_p_z_given_g(beta, g)
		# generate z given p(z|g)
		z_draws = np.random.binomial(1, p_z_given_g)
		# simulate phi and beta
		phi = self._simulate_phi()
		# generate e
		e_draws = np.random.binomial(1, phi[z_draws])
		
		np.savetxt("../input/simulated_data/RIVER/e.txt", e_draws)
		np.savetxt("../input/simulated_data/RIVER/g.txt", g)
		np.savetxt("../input/simulated_data/RIVER/z.txt", z_draws)
		np.savetxt("../input/simulated_data/RIVER/beta.txt", beta)
		np.savetxt("../input/simulated_data/RIVER/phi.txt", phi)
	
	def _simulate_multitask_no_transfer(self):
		tissues = ['brain', 'group1', 'muscle', 'epithelial', 'digestive']
		e = pd.DataFrame()
		z = pd.DataFrame()
		betas = []
		# simulate g data
		g = self._simulate_g()
		alpha_sd = (1.0 / self.lambda_parent) ** (.5)
		# simulate phi using beta distribution (do this step once and use for all tissues)
		phi = self._simulate_phi(self.mu_z0, self.mu_z1, 0.0001, 0.0001)
		for tissue in tissues:
			# simulate beta
			beta = self._simulate_beta_no_parent(alpha_sd)
			betas.append(beta)
			# compute p(z | g, beta)
			p_z_given_g = self._compute_p_z_given_g(beta, g)
			#plt.hist(p_z_given_g)
			# generate z given p(z|g)
			z_draws = np.random.binomial(1, p_z_given_g)
			z[tissue] = z_draws
			# generate e
			e_draws = np.random.binomial(1, phi[z_draws])
			e[tissue] = e_draws
		self.e = e
		self.g = g
		self.z = z
		return e, g, z, pd.DataFrame(betas, index=tissues), phi

	def _simulate_multitask_with_transfer(self):
		tissues = ['brain', 'group1', 'muscle', 'epithelial', 'digestive']
		lambda_hp_children_dict = {'brain': 4, 'group1': 5, 'muscle': 6, 'epithelial': 7, 'digestive': 8}
		e = pd.DataFrame()
		z = pd.DataFrame()
		betas = []
		# simulate g data
		g = self._simulate_g()
		# square root of the variance
		alpha_sd = (1.0 / self.lambda_parent) ** (.5)
		alpha = self._simulate_alpha(alpha_sd)
		# generate phi from Beta distribution
		phi = self._simulate_phi(self.mu_z0, self.mu_z1, 0.0001, 0.0001)
		for tissue in tissues:
			sd = (1.0 / lambda_hp_children_dict[tissue]) ** (.5)
			beta = self._simulate_beta_given_alpha(alpha, sd)
			betas.append(beta)
			# compute p(z | g, beta)
			p_z_given_g = self._compute_p_z_given_g(beta, g)
			#plt.hist(p_z_given_g)
			# generate z given p(z|g)
			z_draws = np.random.binomial(1, p_z_given_g)
			z[tissue] = z_draws
			# generate e
			e_draws = np.random.binomial(1, phi[z_draws])
			e[tissue] = e_draws
		self.e = e
		self.g = g
		self.z = z
		return e, g, z, pd.DataFrame(betas, index=tissues), phi

	def _run(self):
		if self.model_type == 'with_transfer':
			while True:
				e, g, z, betas, phi = self._simulate_multitask_with_transfer()
				x = 0
				for col in z.columns: x+=np.count_nonzero(z[col]) / len(z[col])
				x /= 5
				if x >= 0.1 and x <= 0.9: break
		elif self.model_type == 'without_transfer':
			while True:
				isValid = True
				e, g, z, betas, phi = self._simulate_multitask_no_transfer()
				x = []
				for col in z.columns: 
					x.append(np.count_nonzero(z[col]) / len(z[col]))
				for num in x:
					if num <= 0.01 or num >= 0.99:
						isValid = False
						break
				if isValid:
					break
		else:
			print("ERROR: enter with_transfer or without_transfer")

		# check if directory exists
		if not os.path.isdir(self.output_dir):
			os.makedirs(self.output_dir)
		# write to file
		np.savetxt(self.output_dir + "g.csv", g, delimiter=",")
		e.to_csv(self.output_dir + "e.csv")
		z.to_csv(self.output_dir + "z.csv")
		betas.to_csv(self.output_dir + "betas.csv")
		np.savetxt(self.output_dir + "phi.txt", phi)
'''
if __name__ == "__main__":
	
		Example:
		SimulateData("./simulation_output/", 'with_transfer', 0.4, 0.6, 0.1)
	
	simulate_with_transfer = SimulateData("./hi2_with_transfer/", 'with_transfer', 0.4, 0.6, 0.1)
	simulate_without_transfer = SimulateData("./hi2_without_transfer/", 'without_transfer', 0.4, 0.6, 0.1)

	simulate_with_transfer._run()
	simulate_without_transfer._run()
'''