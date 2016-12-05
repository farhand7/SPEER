#!/usr/bin/env
__author__ = 'farhan_damani'

import numpy as np


'''
    E | Z functions
    - noisyor
    - categorical distributions

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

def estimate_params_noisyor_2_params(E, w, eqtl):
    '''
        Estimate noisy or CPT 
        P(E | RV) and P(E | eqtl)

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

