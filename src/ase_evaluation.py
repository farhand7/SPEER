#!/usr/bin/env
__author__ = 'farhan_damani'

'''
    ASE evaluation using fisher's exact test
    compare allelic imbalance to posteriors
    create bar graphs for each tissue group
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

#ase_path = sys.argv[1]
output_name = sys.argv[1]
train_path = sys.argv[2]

# read in data
train = pd.read_csv(train_path)
#ase = pd.read_csv(ase_path, index_col=(0,1))
#train = pd.read_csv("../src_output/9.19.16.v7run_eqtl_4baselines/train.csv")
ase = pd.read_csv("../input/ase_train.csv", index_col=(0,1))

# remove .notation from genes in training file
train["gene_id"] = [g.split(".")[0] for g in train["gene_id"]]
train.index = [train["subject_id"], train["gene_id"]]
train = train.drop(["subject_id", "gene_id"], axis=1)

def processTissueGroups(tissue_groups_path):
    tissue_groups = {}
    f = open(tissue_groups_path)
    for l in f:
        w = l.strip().split(',')
        group = w[0]
        tissue_groups[group] = []
        for tissue in w[1:]: tissue_groups[group].append(tissue)
    return tissue_groups

# read in tissue groups
tissue_groups = processTissueGroups("../tissue_groups/tissue_groups.txt")

for tissue_group, tissues in tissue_groups.items():
    #ase_df = ase[tissue_groups[tissue_group]].dropna(thresh = 5)
    #ase_df["median"] = ase_df.median(axis=1)
    #ase_df = ase_df.dropna(thresh = 4)
    models = ['posterior','posterior_RIVER', 'posterior_genome_only', 'posterior_genome_only_shared']
    model_comparison_df = pd.concat([train[train["tissue"] == tissue_group][["posterior", "posterior_RIVER", "posterior_genome_only", "posterior_genome_only_shared"]], 
        ase[tissue_groups[tissue_group]].dropna(thresh = 3).median(axis=1)], axis=1).dropna()

    thresholds = [.6, .7, .8, .9]
    results_list = []
    for thresh in thresholds:
        pvalues = []
        for model in models:
            #print (model, "posterior threshold: ", thresh)
            shared_combined_mod = model_comparison_df
            shared_med_90_quant = shared_combined_mod[0].quantile(.9)
            shared_post_90_quant = shared_combined_mod[model].quantile(thresh)

            r = shared_combined_mod[shared_combined_mod[0] > shared_med_90_quant]
            top_left = r[r[model] > shared_post_90_quant].count()[-1]

            r = shared_combined_mod[shared_combined_mod[0] <= shared_med_90_quant]
            top_right = r[r[model] > shared_post_90_quant].count()[-1]

            r = shared_combined_mod[shared_combined_mod[0] > shared_med_90_quant]
            bottom_left = r[r[model] <= shared_post_90_quant].count()[-1]

            r = shared_combined_mod[shared_combined_mod[0] <= shared_med_90_quant]
            bottom_right = r[r[model] <= shared_post_90_quant].count()[-1]

            top = [top_left, top_right]
            bottom = [bottom_left, bottom_right]
            from scipy import stats
            #oddsratio, pvalue = stats.fisher_exact([[265, 1815], [1793, 16920]])
            oddsratio, pvalue = stats.fisher_exact([top, bottom])
            negative_log10_pvalue = -np.log10(pvalue)
            pvalues.extend([negative_log10_pvalue])
        results_list.append(pvalues)
    results = pd.DataFrame(results_list)

    results.columns = ['Multi-task','RIVER', 'Tissue-spec. LR', 'Shared LR']
    results.index = ["60", "70", "80", "90"]
    plt.figure()
    plt.style.use('seaborn-talk')
    ax = results.plot(kind='bar', use_index=True, sort_columns=True, rot=360, figsize=(20,10), legend=(10,10))
    ax.set_xlabel("Posterior percentile threshold")
    ax.set_ylabel("-log(p)")
    ax.legend(loc = 'upper left', borderpad=2, labelspacing = 1)
    #plt.show()
    plt.savefig(output_name+"_"+tissue_group+"_ASE.png")









