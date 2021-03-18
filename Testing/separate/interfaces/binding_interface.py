import numpy as np
import torch 
from torch import nn

import pandas as pd
import os

import sys

sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from Testing.TESTING_procedure_abstract import TEST_PROCEDURE
from BAPTAT_3_binding_class import SEP_BINDING


class TEST_BINDING(TEST_PROCEDURE): 

    def __init__(self, num_features, num_observations, num_dimensions, experiment_name):
        super().__init__(num_features, num_observations, num_dimensions)
        experiment_path = "D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT/Grafics/SeparateBindingRuns/"+experiment_name+'/'
        super().create_trial_directory(experiment_path)
        self.trial_path = self.result_path
        self.BAPTAT = SEP_BINDING()
        print('Initialized test environment.')


    def load_data(self, modify=None, sample_nums=None):
        amc_paths, asf_paths = self.get_data_paths()

        if modify == 'rand' or modify == 'det':
            if modify == 'rand': 
                modification = np.array(range(self.num_observations))
                np.random.shuffle(modification)
                print('Randomly modified order of observed features: ', modification)

            elif modify == 'det': 
                swap_pairs = [(2,4), (6, 10)]
                modification = np.array(range(self.num_observations))
                for i,j in swap_pairs:
                    modification[i], modification[j] = modification[j], modification[i]
                print('Deterministically modified order of observed features: ', modification)

            self.new_order = modification
            self.reorder = torch.tensor([np.where(self.new_order == i)[0][0] 
                                                    for i in range(self.num_observations)])
            modification = torch.tensor(modification, dtype=torch.int64)
            modification = [('rebind', modification)]
        else: 
            modification = None
            self.new_order = None
            self.reorder = None

        data = self.load_data_all(asf_paths, amc_paths, sample_nums, modification)
        
        return data


    def prepare_inference(self, num_frames, model_path, tuning_length, num_tuning_cycles, at_loss_function, at_loss_parameters, at_learning_rate_binding, at_learning_rate_state, at_momentum_binding):
        super().prepare_inference(
            self.BAPTAT, 
            num_frames, 
            model_path, 
            tuning_length, 
            num_tuning_cycles, 
            at_loss_function, 
            at_learning_rate_binding, 
            at_learning_rate_state, 
            at_momentum_binding)

        # set ideal comparision parameters
        # NOTE: ideal matrix is always identity, bc then the FBE and determinant can be calculated => provide reorder
        self.BAPTAT.set_comparison_values(
            torch.Tensor(np.identity(self.num_features)))
        
        info_string = ''
        info_string += f' - modification of binding order: \t{self.new_order}\n\n'
        info_string = self.construct_info_string(info_string, at_loss_parameters)
        info_string += f' - learning rate: \t\t{self.BAPTAT.at_learning_rate}\n'
        info_string += f' - momentum: \t\t\t{self.BAPTAT.bm_momentum}\n'
        
        self.write_to_file(info_string, self.result_path+'parameter_information.txt')
        print('Ready to run AT inference for binding task! \nInitialized parameters with: \n' + info_string)
        

    def evaluate(self, 
        observations, 
        final_predictions, 
        final_binding_matrix,
        final_binding_entries, 
        feature_names, 
        order):

        names = [
            'prediction_errors', 'at_loss_history', 'fbe_history', 'determinante_history', 
            'final_binding_matirx', 'final_binding_neurons_activities']

        results, figures = super().evaluate(self.BAPTAT, observations, final_predictions)
        ## Save figures
        figures += [self.BAPTAT.evaluator.plot_at_losses(results[2], 'History of binding matrix loss (FBE)')]
        figures += [self.BAPTAT.evaluator.plot_at_losses(results[3], 'History of binding matrix determinante')]
        
        if self.num_features != self.num_observations: 
            figures += [self.BAPTAT.evaluator.plot_binding_matrix_nxm(
                final_binding_matrix, 
                feature_names, 
                self.num_observations,
                self.BAPTAT.get_additional_features(),
                'Binding matrix showing relative contribution of observed feature to input feature', 
                order
            )]
            figures += [self.BAPTAT.evaluator.plot_binding_matrix_nxm(
                final_binding_entries, 
                feature_names, 
                self.num_observations,
                self.BAPTAT.get_additional_features(),
                'Binding matrix entries showing contribution of observed feature to input feature', 
                order
            )]
            figures += [self.BAPTAT.evaluator.plot_outcast_gradients(
                self.BAPTAT.get_oc_grads(), 
                feature_names, 
                self.num_observations,
                self.BAPTAT.get_additional_features(),
                'Gradients of outcast line for observed features during inference'
            )]
            names += ['outcat_line_gradients']

        else:
            figures += [self.BAPTAT.evaluator.plot_binding_matrix(
                final_binding_matrix, 
                feature_names, 
                'Binding matrix showing relative contribution of observed feature to input feature', 
                order
            )]
            figures += [self.BAPTAT.evaluator.plot_binding_matrix(
                final_binding_entries, 
                feature_names, 
                'Binding matrix entries showing contribution of observed feature to input feature', 
                order
            )]


        self.save_figures(figures, names)
        self.save_results_to_csv(results, names)

        i = len(results)
        torch.save(final_binding_matrix, self.result_path + names[i] + '.pt')
        i += 1
        torch.save(final_binding_entries, self.result_path + names[i] + '.pt')

        return results



    def run(self, 
        experiment_dir,
        modified,
        sample_nums, 
        model_path, 
        tuning_length, 
        num_tuning_cycles, 
        at_loss_function,
        loss_parameters,
        at_learning_rate_binding, 
        at_learning_rate_state, 
        at_momentum_binding, 
        grad_calculation):

        print('*************************************************************************************')

        experiment_results = []
        res_path = self.trial_path + experiment_dir
        if experiment_dir != "":
            os.mkdir(res_path)
            print('Created directory: '+ res_path)

        data = self.load_data(modify=modified, sample_nums=sample_nums)
        for d in data:
            name = d[0]
            observations = d[1]
            feat_names = d[2]
            num_frames = observations.size()[0]
            self.BAPTAT.set_data_parameters_(
                num_frames, self.num_observations, self.num_features, self.num_dimensions)

            self.result_path = res_path+name+'/'
            os.mkdir(self.result_path)

            new_order = self.new_order
            if new_order is not None and '_' not in name:
                new_order = None
            

            self.prepare_inference(
                num_frames, 
                model_path, 
                tuning_length, 
                num_tuning_cycles, 
                at_loss_function, 
                loss_parameters,
                at_learning_rate_binding, 
                at_learning_rate_state, 
                at_momentum_binding)

            at_final_predictions, final_binding_matrix, final_binding_entries = self.BAPTAT.run_inference(
                observations, grad_calculation, new_order, self.reorder)

            # reorder observations to compare with final predictions
            if new_order is not None:
                observations = self.order(observations, self.reorder)

            res = self.evaluate(observations, at_final_predictions, final_binding_matrix, final_binding_entries, feat_names, new_order)
            print('Evaluated current run.')

            super().terminate()
            print('Terminated current run.')

            experiment_results += [[name, res, final_binding_matrix, final_binding_entries]]

        return experiment_results
      
        