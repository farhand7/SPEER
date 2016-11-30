#!/usr/bin/env
__author__ = 'farhan_damani'

import numpy as np


'''
    Standalone Naive Bayes Model for P(E | Z)

    sample code to use model:

    param_dict = _build_global_table(NUM_FEATURES) # initialize dictionary
    updated_param_dict = _parameter_update(param_dict, w_j, E) # estimate parameters with initialized w values, expression matrix
    log_p = _log_probability_e_given_z(parameters, E_i, j) # compute log P(E|Z)
'''
import math


def estimate_params(w, E):
    ct = np.zeros((2,2))
    #smoothing = 10
    ct[0, 0] = np.sum((1.0 - E) * (1.0 - w))
    ct[0, 1] = np.sum((1.0 - E) * w)
    ct[1, 0] = np.sum(E * (1.0 - w))
    ct[1, 1] = np.sum(E * w)
    ct += 10
    ct = ct / np.sum(ct, axis = 0)
    return ct

def log_prob(E, z, ct):
    '''
        Compute log probability of each example

    '''  
    return np.log(ct[0, z] * (1.0 - E) + ct[1, z] * E)

def log_prob_with_indel_observed(E, z, I, ct):
    '''
        Compute log probability of each sample
    '''
    '''
    return np.log(ct[0, z, 0] * (1.0 - E) * (1.0 - I) + ct[0, z, 1] * (1.0 - 
        E) * I + ct[1, z, 0] * E * (1.0 - I) + ct[1, z, 1] * E * I)
    '''
    return np.log(ct[0,z,1] * (1.0 - E) * I + ct[1,z,1] * E * I + ct[0,z,0] * (1.0 - E) * (1.0 - I) + ct[1,z,0] * E * (1.0 - I))
    #return np.log(ct[0,z,1] * (1.0 - E) + ct[1,z,1] * E)

def log_prob_with_splicing(E, z, S, ct):
    '''
        Compute log probability of each sample
        log p(E, S | z)
    '''

    return np.log(ct[0,0,z] * (1.0 - E) * (1.0 - S) + ct[1,1,z] * E * S + ct[0,1,z] * (1.0 - E) * S + ct[1,0,z] * E * (1.0 - S))


def estimate_params_noisyor_2_preds(w, E, eqtl):
    '''
        estimate noisy OR CPT 
        P(E | Z, eqtl)
    '''
    # first, let's compute
    ct = np.zeros(2)
    ct[0] = np.sum(E * w)
    ct[1] = np.sum(E * eqtl)
    return ct

def estimate_params_with_splicing(w, E, S):
    '''
        estimate table probabilities for
        P(E, S | Z)
    '''
    ct = np.zeros((2,2,2))
    ct[0, 0, 0] = np.sum((1.0 - E) * (1.0 - S) * (1.0 - w))
    ct[0, 1, 0] = np.sum((1.0 - E) * S * (1.0 - w))
    ct[1, 0, 0] = np.sum(E * (1.0 - S) * (1.0 - w))
    ct[1, 1, 0] = np.sum(E * S * (1.0 - w))

    ct[0, 0, 1] = np.sum((1.0 - E) * (1.0 - S) * w)
    ct[0, 1, 1] = np.sum((1.0 - E) * S * w)
    ct[1, 0, 1] = np.sum(E * (1.0 - S) * w)
    ct[1, 1, 1] = np.sum(E * S * w)
    ct += 10

    x = np.sum(ct, axis=(0,1)) # sum over E, AS - 1st and 2nd dimension
    ct[:,:,0] /= x[0]
    ct[:,:,1] /= x[1]
    return ct


def estimate_params_with_indel_observed(w, E, I):
    '''
        Estimate table probabilities
        P(E | Z, G_indel)
    '''
    ct = np.zeros((2,2,2))
    ct[0, 0, 0] = np.sum((1.0 - E) * (1.0 - w) * (1.0 - I))
    ct[0, 1, 0] = np.sum((1.0 - E) * w * (1.0 - I))
    ct[1, 0, 0] = np.sum(E * (1.0 - w) * (1.0 - I))
    ct[1, 1, 0] = np.sum(E * w * (1.0 - I))

    ct[0, 0, 1] = np.sum((1.0 - E) * (1.0 - w) * I)
    ct[0, 1, 1] = np.sum((1.0 - E) * w * I)
    ct[1, 0, 1] = np.sum(E * (1.0 - w) * I)
    ct[1, 1, 1] = np.sum(E * w * I)
    ct += 10
    x = np.sum(ct, axis=0) # sum over E
    ct[:, 0, 0] /= x[0, 0]
    ct[:, 1, 0] /= x[1, 0]
    ct[:, 0, 1] /= x[0, 1]
    ct[:, 1, 1] /= x[1, 1]

    return ct

def estimate_params_with_indel(E, w):
    '''
        Estimate table probabilities
        P(E | Z_snv, Z_indel
    '''
    ct = np.zeros((2,2,2))
    ct[0, 0, 0] = np.sum((1.0 - E) * w[0,0])
    ct[0, 1, 0] = np.sum((1.0 - E) * w[1,0])
    ct[1, 0, 0] = np.sum(E * w[0,0])
    ct[1, 1, 0] = np.sum(E * w[1,0])

    ct[0, 0, 1] = np.sum((1.0 - E) * w[0,1])
    ct[0, 1, 1] = np.sum((1.0 - E) * w[1,1])
    ct[1, 0, 1] = np.sum(E * w[0,1])
    ct[1, 1, 1] = np.sum(E * w[1,1])
    ct += 10
    x = np.sum(ct, axis=0) # sum over E
    ct[:, 0, 0] /= x[0, 0]
    ct[:, 1, 0] /= x[1, 0]
    ct[:, 0, 1] /= x[0, 1]
    ct[:, 1, 1] /= x[1, 1]

    return ct

def estimate_params_noisyor_3_params(E, w, eqtl, sv):
    '''
        Estimate noisy or CPT 
        P(Z_snv | RV) and P(Z_eqtl | eqtl)

    '''
    ct = np.zeros(3)
    # p(z_snv = 1 | rv = 1)
    ct[0] = np.sum(E*w) / float(np.sum(w))
    # p(z_eqtl = 1 | eqtl = 1)
    ct[1] = np.sum(E*eqtl) / float(np.sum(eqtl))

    ct[2] = np.sum(E*sv) / float(np.sum(sv))
    return ct


def estimate_params_noisyor_1_param(E, w):
    ct = np.zeros(1)
    ct[1] = np.sum(E*w) / float(np.sum(w))
    return ct


def estimate_params_noisyor_2_params(E, w, eqtl):
    '''
        Estimate noisy or CPT 
        P(Z_snv | RV) and P(Z_eqtl | eqtl)

    '''
    ct = np.zeros(2)
    # p(z_snv = 1 | rv = 1)
    ct[0] = np.sum(E*w) / float(np.sum(w))
    # p(z_eqtl = 1 | eqtl = 1)
    ct[1] = np.sum(E*eqtl) / float(np.sum(eqtl))
    return ct

def log_prob_noisyor_2_params(E, z, eqtl, ct):

    if z == 1:
        data = E * z * eqtl
        p_e_1_given_z1_eqtl1 = 1.0 * data  - (1.0 - ct[0]) * (1.0 - ct[1]) * data
        
        data = (1.0 - E) * z * eqtl
        p_e_0_given_z1_eqtl1 = (1.0 - ct[0]) * (1.0 - ct[1]) * data

        data = E * z * (1.0 - eqtl)
        p_e_1_given_z1_eqtl0 = ct[0] * data
        
        data = (1.0 - E) * z * (1.0 - eqtl)
        p_e_0_given_z1_eqtl0 = 1.0 * data - ct[0] * data
        p = p_e_1_given_z1_eqtl1 + p_e_0_given_z1_eqtl1 + p_e_1_given_z1_eqtl0 + p_e_0_given_z1_eqtl0
        return np.log(p)
    else:
        # P(e = 1 | z = 0, eqtl = 1) = p(e = 1) | eqtl = 1)
        data = E * (1.0 - z) * eqtl
        p_e_1_given_z0_eqtl1 = ct[1] * data

        data = (1.0 - E) * (1.0 - z) * eqtl
        # P(e = 0 | z = 0, eqtl = 1) = p(e = 0 | eqtl = 1)
        p_e_0_given_z0_eqtl1 = 1.0 * data - ct[1] * data

        data = E * (1.0 - z) * (1.0 - eqtl)
        # P(e = 1 | z = 0, eqtl = 0)
        p_e_1_given_z0_eqtl0 = .000001 * data

        data = (1.0 - E) * (1.0 - z) * (1.0 - eqtl)
        # P(e = 0 | z = 0, eqtl = 0)
        p_e_0_given_z0_eqtl0 = .999999 * data
        p = p_e_1_given_z0_eqtl1 + p_e_0_given_z0_eqtl1 + p_e_1_given_z0_eqtl0 + p_e_0_given_z0_eqtl0
        return np.log(p)

