import numpy as np
import torch 
from torch import nn

import pandas as pd
import os

import sys

sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from Testing.TESTING_procedure_abstract import TEST_PROCEDURE
from BAPTAT_3_translation_class import SEP_TRANSLATION
from BinAndPerspTaking.perspective_taking import Perspective_Taker


class TEST_TRANSLATION(TEST_PROCEDURE): 

    def __init__(self, num_features, num_observations, num_dimensions, experiment_name):
        super().__init__(num_features, num_observations, num_dimensions)
        experiment_path = "D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT/Grafics/SeparateTranslationRuns/"+experiment_name+'/'
        super().create_trial_directory(experiment_path)
        self.trial_path = self.result_path
        self.BAPTAT = SEP_TRANSLATION()
        self.PERSP_TAKER = Perspective_Taker(num_observations, num_dimensions, False, False)
        print('Initialized test environment.')


    def load_data(self, modify=None, trans_range=None, sample_nums=None):
        amc_paths, asf_paths = self.get_data_paths()

        if modify == 'rand' or modify == 'det':
            if modify == 'rand': 
                lower_bound = trans_range[0] * torch.ones(self.num_dimesions)
                upper_bound = trans_range[1] * torch.ones(self.num_dimesions)
                range_size = upper_bound - lower_bound
                modification = torch.mul(torch.rand(self.num_dimensions), range_size)
                print(f'Randomly modified translation of observed features: {modification}')

            elif modify == 'det': 
                modification = torch.Tensor([3.2, -2.6, 0.4])
                print(f'Deterministically modified translation of observed features: {modification}')

            self.new_translation = modification
            self.retranslate = self.PERSP_TAKER.inverse_translation_bias(modification)

            modification = [('translate', modification)]
        else: 
            modification = None
            self.new_rotation = None
            self.retranslate = torch.zeros(self.num_dimensions)


        data = self.load_data_all(asf_paths, amc_paths, sample_nums, modification)
        
        return data


    def prepare_inference(self, 
        current_translation, 
        num_frames, 
        model_path, 
        tuning_length, 
        num_tuning_cycles, 
        at_loss_function, 
        at_loss_parameters, 
        at_learning_rate_translation, 
        at_learning_rate_state, 
        at_momentum_translation):


        super().prepare_inference(
            self.BAPTAT, 
            num_frames, 
            model_path, 
            tuning_length, 
            num_tuning_cycles, 
            at_loss_function, 
            at_learning_rate_translation, 
            at_learning_rate_state, 
            at_momentum_translation)

        # set ideal comparision parameters
        if current_translation is not None: 
            retrans = self.retranslate
        else: 
            retrans = torch.zeros(self.num_dimensions)

        self.BAPTAT.set_comparison_values(retrans)
        
        info_string = ''
        info_string += f' - modification of body translation: {self.new_translation if current_translation is not None else None}\n'
        info_string += f' - optimally infered translation: \n\t{retrans}\n\n'
        info_string = self.construct_info_string(info_string, at_loss_parameters)
        info_string += f' - learning rate: \t\t{self.BAPTAT.at_learning_rate}\n'
        info_string += f' - momentum: \t\t\t{self.BAPTAT.c_momentum}\n'
        
        self.write_to_file(info_string, self.result_path+'parameter_information.txt')
        print('Ready to run AT inference for rotation task! \nInitialized parameters with: \n' + info_string)
        

    def evaluate(self, 
        observations, 
        final_predictions, 
        final_translation_values, 
        feature_names, 
        translation):

        results, figures = super().evaluate(self.BAPTAT, observations, final_predictions)
        ## Save figures
        figures += [self.BAPTAT.evaluator.plot_at_losses(results[2], 'History of translation loss (MSE)')]


        names = ['prediction_errors', 'at_loss_history', 'transba_loss_history', 'final_translation_values']

        self.save_figures(figures, names)
        self.save_results_to_csv(results, names)

        i = len(results)
        torch.save(final_translation_values, self.result_path + names[i] + '.pt')

        return results



    def run(self, 
        experiment_dir,
        modified,
        translation_range,
        sample_nums, 
        model_path, 
        tuning_length, 
        num_tuning_cycles, 
        at_loss_function,
        loss_parameters,
        at_learning_rate_translation, 
        at_learning_rate_state, 
        at_momentum_translation,
        grad_calculation):

        print('*************************************************************************************')

        experiment_results = []
        res_path = self.trial_path + experiment_dir
        if experiment_dir != "":
            os.mkdir(res_path)
            print('Created directory: '+ res_path)


        data = self.load_data(
            modify=modified, 
            trans_range=translation_range,
            sample_nums=sample_nums)


        for d in data:
            name = d[0]
            observations = d[1]
            feat_names = d[2]
            num_frames = observations.size()[0]
            self.BAPTAT.set_data_parameters_(num_frames, self.num_observations, self.num_features, self.num_dimensions)

            self.result_path = res_path+name+'/'
            os.mkdir(self.result_path)

            new_translation = self.new_translation
            if new_translation is not None and '_' not in name:
                new_translation = None
                        
            
            self.prepare_inference(
                new_translation,
                num_frames, 
                model_path, 
                tuning_length, 
                num_tuning_cycles, 
                at_loss_function, 
                loss_parameters,
                at_learning_rate_translation, 
                at_learning_rate_state, 
                at_momentum_translation)

            at_final_inputs, at_final_predictions, final_translation_values = self.BAPTAT.run_inference(observations, grad_calculation)

            self.render(at_final_inputs.view(num_frames, self.num_features, self.num_dimensions))
            self.render(at_final_predictions.view(num_frames, self.num_features, self.num_dimensions))
            self.save_results_to_pt([at_final_inputs], ['final_inputs'])

            # retranslate observations to compare with final predictions 
            if new_translation is not None:
                self.PERSP_TAKER.translate(observations, self.retranslate)
            

            res = self.evaluate(
                observations, 
                at_final_predictions, 
                final_translation_values, 
                feat_names, 
                new_translation)
            print('Evaluated current run.')

            super().terminate()
            print('Terminated current run.')

            experiment_results += [[name, res, final_translation_values]]

        return experiment_results
      