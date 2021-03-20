import pandas as pd   
import matplotlib.pyplot as plt
import seaborn as sns
import abc 
from abc import ABC, abstractmethod

import torch 
import os
import sys
sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')

class TEST_STATISTICS(ABC): 

    def __init__(self, num_features, num_observations, num_dimensions): 
        self.num_features = num_features
        self.num_observations = num_observations
        self.num_dimensions = num_dimensions

        print('Initialized statistical evaluator.')


    ########### BASIC STATISTICS ###########
    def get_basic_statistics(self, data):
        ...


    def plot_histories(self, history_dfs, path, variation, results, titles): 
        for i in range(len(history_dfs)):
            dt = history_dfs[i]
            fig, ax = plt.subplots(figsize = (11,7))
            ax = sns.lineplot(
                x="run", 
                y=results[i], 
                hue=variation, 
                data = dt, 
                palette="Reds"
            )
            ax.set_title(
                titles[i], 
                fontdict= { 'fontsize': 20, 'fontweight':'bold'})
            plt.setp(ax.get_legend().get_texts(), fontsize='15') 
            plt.setp(ax.get_legend().get_title(), fontsize='18')
            ax.set_ylabel(results[i], fontsize=18)
            ax.set_xlabel('run', fontsize=18)

            fig.savefig(path+results[i]+'.png')  


    def plot_value_comparisons(self, history_dfs, path, variation, results, titles):
        for i in range(len(history_dfs)):
            dt = history_dfs[i]
            fig, ax = plt.subplots(figsize = (11,7))
            ax = sns.boxplot(
                x=variation,
                y=results[i],
                data=dt
            )
            ax.set_title(
                titles[i], 
                fontdict= { 'fontsize': 20, 'fontweight':'bold'})
            ax.set_ylabel(results[i], fontsize=18)
            ax.set_xlabel('run', fontsize=18)

            fig.savefig(path+results[i]+'comp.png')  


    def load_csvresults_to_dataframe(self, experiment_path, variation, variation_values, samples, results):
        dfs = []
        for result in results:
            value_dfs = []
            for val in variation_values:
                val_path = experiment_path+variation+val+'/'
                sample_dfs = []
                for sample in samples:
                    sdf = pd.read_csv(val_path+sample+'/'+result+'.csv')
                    sdf.columns = ['run', result]
                    sample_dfs += [sdf]
                
                vdf = pd.concat(sample_dfs).sort_values('run').assign(variation=val)
                value_dfs += [vdf]

            df = pd.concat(value_dfs).sort_values('run')
            dfs += [df]
        
        return dfs



    ########### STATISTICAL TESTS ###########


    ####### CORRELATION TESTS


    ####### STATIONARY TESTS


    ####### PARAMETRIC STATISTICAL HYPOTHESIS TEST

    ##### ANOVA





