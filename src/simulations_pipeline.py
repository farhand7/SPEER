'''
	Complete pipeline to generate ROC curves using simulated data.

	Repeat N times (where N is number of simulations):

		1. Simulate data using one of the six parameter settings described in the paper
			'tied_stronger'
			'tied_equal'
			'tied_weaker'
			'independent_stronger'
			'independent_equal'
			'independent_weaker'
		
		2. Create a process object using simulated data
			the process object creates the core data structure that is used when running SPEER.

		3. Run SPEER (network.py)

		4. Run various benchmarks for comparison
			SPEER without transfer (network.py)
			RIVER (RIVER.py)
			tissue specific genome only (benchmark_posteriors.py)
			shared tissue genome only (benchmark_posteriors.py)

		5. Save averaged false positive rate and true positive rate across various thresholds in order to generate plots

	Running this script via command line:
		python simulations_pipeline.py <simulation setting> <number of simulations>
		example: python simulations_pipeline.py tied_stronger 75
'''

import pandas as pd
import numpy as np

import ase_evaluation as ae
import bootstrap as btstrp
import cross_validation as cv
import logistic_regression as lr
import naive_bayes as nb
import network as ntwk
import process as prcs
import simulate_data as sim
from scipy import interp

import matplotlib.pyplot as plt
import seaborn as sns
import benchmark_posteriors as bnchmk
import sklearn
import RIVER as river
import pickle
import sys

setting = str(sys.argv[1])
print(setting)
num_sims = int(sys.argv[2])


if setting == 'tied_stronger':
	phi_init = [0.4, 0.6]
	Lambda = 0.01
elif setting == 'tied_equal':
	phi_init = [0.3, 0.7]
	Lambda = 0.01
elif setting == 'tied_weaker':
	phi_init = [0.4, 0.6]
	Lambda = 0.1
elif setting == 'independent_stronger':
	phi_init = [0.4, 0.6]
	Lambda = 0.01
elif setting == 'independent_equal':
	phi_init = [0.3, 0.7]
	Lambda = 0.01
elif setting == 'independent_weaker':
	phi_init = [0.4, 0.6]
	Lambda = 0.1
else:
	print("ERROR: incorrect setting usage.")
	sys.exit()

if 'tied' in setting:
	model_type = 'with_transfer'
elif 'independent' in setting:
	model_type = 'without_transfer'
else:
	print("ERROR: incorrect model type.")
	sys.exit()

test_data = []
models = ['SPEER', 'SPEER without transfer', 'RIVER', 'tissue specific genome only', 'shared tissue genome only']
fpr, tpr, auc, mean_tpr, mean_fpr, mean_auc = {}, {}, {}, {}, {}, {}

for model in models:
	mean_tpr[model] = {}
	mean_fpr[model] = {}
	mean_auc[model] = {}
for i in range(num_sims):
	# generate simulated data
	data_dir = setting + '_simulated_data/'
	#s = sim.SimulateData("./test_output/", 'with_transfer', 0.4, 0.6, 0.01)
	s = sim.SimulateData(data_dir, model_type, phi_init[0], phi_init[1], Lambda)
	s._run()
	
	# create a process object
	p = prcs.Process(data_dir, 0.1)
	p._process_simulated_data()

	print("simulated and processed data...")
	# run SPEER
	n = ntwk.Network(p.train_list, p.test_list, p.tissues, p.genomic_features, 
				 with_transfer=True, output_dir="SPEER_output", 
				 lambda_hp_parent = None,
				 lambda_hp_children_dict = None,
				 e_distribution = 'cat')
	train_list, test_list, beta_parent, beta_children, phi = n.run()
	
	print("completed SPEER...")
	# run SPEER without transfer
	lambda_hp_children_dict = {'brain': 0.01, 'group1': 0.01, 'muscle': 0.01, 'epithelial': 0.01, 'digestive': 0.01}
	n = ntwk.Network(train_list, test_list, p.tissues, p.genomic_features, 
					 with_transfer=False, output_dir="SPEER_output", 
					 lambda_hp_parent = None, 
					 lambda_hp_children_dict = lambda_hp_children_dict, 
					 e_distribution = 'cat')
	train_list, test_list, beta_parent, beta_children, phi = n.run()
	
	print("completed SPEER without transfer...")
	n = river.River(train_list, test_list, p.genomic_features, output_dir='RIVER_output')
	train_list, test_list, beta_parent_river, beta_children_river, phi_river = n.run()
	
	print("completed RIVER....")
	# add benchmarks 
	bn = bnchmk.BenchmarkPosteriors(train_list, test_list, p.genomic_features)
	train_list, test_list = bn.fit_models()

	print("completed benchmarks...")
	
	for model in models:
		auc = 0
		fpr_local, tpr_local, auc_local = {}, {}, {}
		# for each tissue
		for j in range(len(test_list)):
			fpr_local[j], tpr_local[j], _ = sklearn.metrics.roc_curve(test_list[j]["z_label"], test_list[j][model])
			auc_local[j] = sklearn.metrics.roc_auc_score(test_list[j]["z_label"], test_list[j][model])
		mean_tpr[model][i] = 0.0
		mean_fpr[model][i] = np.linspace(0,1,100)
		for j in range(len(test_list)):
			mean_tpr[model][i] += interp(mean_fpr[model][i], fpr_local[j], tpr_local[j])
			mean_tpr[model][i][0] = 0.0
		mean_tpr[model][i] /= len(test_list)
		mean_tpr[model][i][-1] = 1.0
		mean_auc[model][i] = sklearn.metrics.auc(mean_fpr[model][i], mean_tpr[model][i])
	print(i)


with open('mean_tpr_' + setting + '.obj', 'wb') as fp:
	pickle.dump(mean_tpr, fp)
with open('mean_fpr_' + setting + '.obj', 'wb') as fp:
	pickle.dump(mean_fpr, fp)
with open('mean_auc_' + setting + '.obj', 'wb') as fp:
	pickle.dump(mean_auc, fp)