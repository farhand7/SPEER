#!/usr/bin/env
__author__ = 'farhan_damani'

'''
	Evaluate a set of lambda parameter values on L2-regularized logistic regression using test-set AUC.


'''
import pandas as pd
import logistic_regression as lr
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn import linear_model

e = pd.read_csv("./expression.short.csv", index_col=(0,1))
g_generic = pd.read_csv("./g_train.csv", index_col=(0,1))
g_tissuespec = pd.read_csv("../../../../../data/annotations/annotation_train/tissue_spec_annotations_train_max.csv",index_col=(0,1))
train = pd.concat([g_generic, g_tissuespec, e], axis=1)

lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]

annotation_cols = list(g_generic.columns)
annotation_cols.extend(list(g_tissuespec.columns))
train["labels"] = sklearn.preprocessing.binarize(np.abs(train["median"]).reshape(-1,1), threshold=1.5)
G = train[annotation_cols]
G.insert(0,"intercept",1)
E = train["labels"]
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(G, E, test_size=0.33, random_state=42)

fpr, tpr, auc, betas = dict(), dict(), dict(), dict()
for i in range(len(lambdas)):
    lambda_hp = lambdas[i]
    beta = lr.sgd(x_train, y_train, np.ones(G.shape[1]), np.zeros(G.shape[1]), lambda_hp)
    betas[lambda_hp] = beta
    fpr[lambda_hp], tpr[lambdas[i]], _ = sklearn.metrics.roc_curve(y_test, np.exp(lr.log_prob(x_test, beta)))
    auc[lambda_hp] = sklearn.metrics.auc(fpr[lambda_hp], tpr[lambda_hp])
    print(auc[lambda_hp])
import pdb; pdb.set_trace()