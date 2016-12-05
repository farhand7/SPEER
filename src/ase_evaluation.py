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
from scipy import stats


class ASE_Evaluation:

    def __init__(self, train_list, ase_path, tissue_groups, models=None, posterior_percentile_thresholds=[.6, .7, .8, .9])
        '''
            train_list : list
                List of tissue-specific training matrices

            ase_path : str
                Path to ASE data

            tissue_groups : dictionary
                Keys - tissue group names
                Values - list of tissues for each tissue group
            
            models : list, default : ['SPEER', 'tissue specific genome only', 'shared tissue genome only']
                Models used for ASE comparison 
        '''

        self.train_list = train_list
        self.ase_path = ase_path
        self.tissue_groups = tissue_groups
        self.thresholds = thresholds
        
        if models == None:
            self.models = ['SPEER', 'tissue specific genome only', 'shared tissue genome only']
        else:
            self.models = models

    def _append_allelic_ratios(self):
        for i,train in enumerate(self.train_list):
            tissue = train["tissue"].iloc[0]
            train["gene_id"] = [g.split(".")[0] for g in train["gene_id"]]
            train.index = [train["subject_id"], train["gene_id"]]
            if tissue == 'brain':
                allelic_ratio = ase[self.tissue_groups[tissue]].dropna(thresh = 3).median(axis=1)
            elif tissue == 'group1':
                allelic_ratio = ase[self.tissue_groups[tissue]].dropna(thresh = 4).median(axis=1)
            else:
                allelic_ratio = ase[self.tissue_groups[tissue]].dropna(thresh = 2).median(axis=1)

            allelic_ratio.name = 'median_ase'
            self.train_list[i] = pd.concat([train, allelic_ratio], axis=1).dropna()

        for train in self.train_list:
            thresh = train["median_ase"].quantile(.9)
            rate = len(train[train["median_ase"] > thresh]) / len(train)
            train["average"] = rate

    def _compute_fishers_exact_test(self):

        list(tissue_groups.keys())
        for i,train in enumerate(self.train_list):
            tissue = train["tissue"].iloc[0]
            results_list = []
            for thresh in self.thresholds:
                pvalues = []
                for model in self.models:
                    shared_combined_mod = train
                    #shared_combined_mod.columns = ['SPEER', 'RIVER', 'tissue specific genome only', 'shared tissue genome only', 'expr_label', 'median_ase', 'average']
                    shared_combined_mod.columns = ['SPEER', 'RIVER', 'tissue specific genome only', 'shared tissue genome only', 'expr_label', 'median_ase', 'average']
                    
                    shared_med_90_quant = shared_combined_mod['median_ase'].quantile(.90)
                    shared_post_90_quant = shared_combined_mod[model].quantile(thresh)

                    r = shared_combined_mod[shared_combined_mod['median_ase'] > shared_med_90_quant]
                    
                    top_left = r[r[model] > shared_post_90_quant].count()[-1]
                    r = shared_combined_mod[shared_combined_mod['median_ase'] <= shared_med_90_quant]
                    top_right = r[r[model] > shared_post_90_quant].count()[-1]
                    r = shared_combined_mod[shared_combined_mod['median_ase'] > shared_med_90_quant]
                    bottom_left = r[r[model] <= shared_post_90_quant].count()[-1]

                    r = shared_combined_mod[shared_combined_mod['median_ase'] <= shared_med_90_quant]
                    bottom_right = r[r[model] <= shared_post_90_quant].count()[-1]

                    top = [top_left, top_right]
                    bottom = [bottom_left, bottom_right]
                    oddsratio, pvalue = stats.fisher_exact([top, bottom])
                    negative_log10_pvalue = -np.log10(pvalue)
                    pvalues.extend([negative_log10_pvalue])
                results_list.append(pvalues)
            results = pd.DataFrame(results_list)
            #results.columns = list(train.columns[0:4])
            results.columns = models
            results.index = ["60", "70", "80", "90"]
            results.index.name

            fig = plt.figure(2)
            fig.set_size_inches(3, 3)
            #plt.tight_layout()
            #plt.style.use('seaborn-talk')
            sns.set_context("paper")
            sns.set_palette("deep")
            sns.set(font='serif')
            sns.set_style("white", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})

            #sns.palplot(sns.color_palette("Set1", n_colors=8, desat=.5))

            ax = results.plot(kind='bar', use_index=True, sort_columns=True, rot=360, figsize=(15,10))
            #ax.hlines(3.5, -0.5, 4.5, linestyles='--', linewidth=1)
            ax.set_xlabel("Posterior percentile threshold", fontsize=30)
            ax.set_ylabel("-log$_{10}$(P-value)", fontsize=30)
            #ax.set_ylim(0,20)
            ax.legend(loc = 'upper left', borderpad=2, labelspacing = 1, fontsize=17)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20)
            #plt.figtext(0.5, 0.9, tissues[i], fontsize=20, ha='center')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #plt.title(tissue, fontsize=30)
            #plt.text(-0.8, 31, 'A', fontsize=35, horizontalalignment='center')
            #plt.savefig('ase_'+str(tissue)+'.png', format='png', dpi=200) # This does, too
            plt.savefig('ASE_aggregate_1_75_no_river', format='png', dpi=300) # This does, too
            
            #plt.tight_layout()
            break








