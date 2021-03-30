import pandas as pd   
import matplotlib.pyplot as plt
import seaborn as sns
import abc 
from abc import ABC, abstractmethod

import torch 
import os
import sys
sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')

class TEST_STATISTICS(): 

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
            fig, ax = plt.subplots(figsize = (40,20))
            ax = sns.lineplot(
                x="run", 
                y=results[i], 
                hue=variation, 
                data = dt 
                # ,palette="Reds"
            )
            ax.set_title(
                titles[i], 
                fontdict= { 'fontsize': 20, 'fontweight':'bold'})
            plt.setp(ax.get_legend().get_texts(), fontsize='15') 
            plt.setp(ax.get_legend().get_title(), fontsize='18')
            ax.set_ylabel(results[i], fontsize=18)
            ax.set_xlabel('run', fontsize=18)

            fig.savefig(path+results[i]+'.png')  
        
        plt.close('all')


    def plot_value_comparisons(self, history_dfs, path, variation, results, titles):
        for i in range(len(history_dfs)):
            dt = history_dfs[i]
            fig, ax = plt.subplots(figsize = (40,20))
            ax = sns.boxplot(
                x=variation,
                y=results[i],
                data=dt
            )
            ax.set_title(
                titles[i], 
                fontdict= { 'fontsize': 20, 'fontweight':'bold'})
            ax.set_ylabel(results[i], fontsize=18)
            ax.set_xlabel(variation, fontsize=18)

            fig.savefig(path+results[i]+'comp.png')  

        plt.close('all')


    def load_csvresults_to_dataframe(self, experiment_path, variation, variation_values, samples, results):
        dfs = []
        for result in results:
            value_dfs = []
            for val in variation_values:
                val = f'{val}'
                val_path = experiment_path+variation+'_'+val+'/'
                sample_dfs = []
                for sample in samples:
                    sdf = pd.read_csv(val_path+sample+'/'+result+'.csv')
                    sdf.columns = ['run', result]
                    sample_dfs += [sdf]
                
                vdf = pd.concat(sample_dfs).sort_values('run').assign(variation=val)
                vdf.columns = ['run', result, variation]
                value_dfs += [vdf]

            df = pd.concat(value_dfs).sort_values('run')
            dfs += [df]
        
        return dfs



    ########### STATISTICAL TESTS ###########


    ####### CORRELATION TESTS


    ####### STATIONARY TESTS


    ####### PARAMETRIC STATISTICAL HYPOTHESIS TEST

    ##### ANOVA


def main(): 
    # set the following parameters
    num_observations = 15
    num_input_features = 15
    num_dimensions = 3
    stats = TEST_STATISTICS(num_input_features, num_observations, num_dimensions)
    
    # path = 'D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT/Grafics/GestaltRuns/compare_gestalten_trans_rot_bin/2021_Mar_29-21_45_10/b_r_t_'
    # param = 'dimensions'
    # param_vals = [3, 6, 7]
    # samples_nomod = ['S35T07', 'S08T02', 'S06T01', 'S05T01']
    # samples_mod = ['S35T07_rebinded_qrotated_translated', 'S08T02_rebinded_qrotated_translated', 'S06T01_rebinded_qrotated_translated', 'S05T01_rebinded_qrotated_translated']
    # result_names = ['prediction_errors', 'at_loss_history', 'determinante_history', 'fbe_history', 'rotmat_loss_history', 'rotval_loss_history', 'transba_loss_history']

    path = 'D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT/Grafics/GestaltRuns/compare_gestalten_rotation/2021_Mar_29-20_31_47/r_'
    param = 'dimensions'
    param_vals = [3, 6, 7]
    samples_nomod = ['S35T07', 'S06T01', 'S05T01']
    samples_mod = ['S35T07_qrotated', 'S06T01_qrotated', 'S05T01_qrotated']
    result_names = ['prediction_errors', 'at_loss_history', 'rotval_loss_history']


    # No modification
    dfs_nomod = stats.load_csvresults_to_dataframe(
        path, 
        param, 
        param_vals, 
        samples_nomod, 
        result_names
    )

    stats.plot_histories(
            dfs_nomod, 
            path, 
            param, 
            result_names, 
            result_names
        )

    stats.plot_value_comparisons(
            dfs_nomod, 
            path, 
            param, 
            result_names, 
            result_names
        )



    # With modification
    dfs_mod = stats.load_csvresults_to_dataframe(
        path, 
        param, 
        param_vals, 
        samples_mod, 
        result_names
    )

    
    stats.plot_histories(
            dfs_mod, 
            path, 
            param, 
            result_names, 
            result_names
        )

    stats.plot_value_comparisons(
            dfs_mod, 
            path, 
            param, 
            result_names, 
            result_names
        )


if __name__ == "__main__":
    main()


