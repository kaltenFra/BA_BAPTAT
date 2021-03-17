import numpy as np
import torch 
from torch import nn

import pandas as pd
import os

import sys

sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from Testing.TESTING_procedure_abstract import TEST_PROCEDURE
from BAPTAT_3_rotation_class import SEP_ROTATION
from BinAndPerspTaking.perspective_taking import Perspective_Taker


class TEST_ROTATION(TEST_PROCEDURE): 

    def __init__(self, num_features, num_observations, num_dimensions, experiment_name):
        super().__init__(num_features, num_observations, num_dimensions)
        experiment_path = "D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT/Grafics/SeparateRotationRuns/"+experiment_name+'/'
        super().create_trial_directory(experiment_path)
        self.trial_path = self.result_path
        self.BAPTAT = SEP_ROTATION()
        self.PERSP_TAKER = Perspective_Taker(num_observations, num_dimensions, False, False)
        print('Initialized test environment.')


    def load_data(self, modify=None, rotation_type=None, sample_nums=None):
        amc_paths, asf_paths = self.get_data_paths()

        if modify == 'rand' or modify == 'det':
            if modify == 'rand': 
                if rotation_type == 'qrotate':
                    modification = torch.rand(4) 
                elif rotation_type == 'eulrotate': 
                    modification = torch.rand(3)*360

                print('Randomly modified rotation of observed features: {rotation_type} by {modification}')

            elif modify == 'det': 
                modification = torch.Tensor([0.5,0.4,0.3,0.2]) 
                if rotation_type == 'eulrotate': 
                    modification = self.PERSP_TAKER.qeuler(modification,'xyz').view(3,1)
                
                print(f'Deterministically modified rotation of observed features: {rotation_type} by {modification}')

            self.new_rotation = modification
            if rotation_type == 'qrotate':
                self.rerotate = self.PERSP_TAKER.inverse_rotation_quaternion(modification)
            elif rotation_type == 'eulrotate': 
                self.rerotate = self.PERSP_TAKER.inverse_rotation_angles(modification)

            modification = [(rotation_type, modification)]
        else: 
            modification = None
            self.new_rotation = None
            self.rerotate = None

        data = self.load_data_all(asf_paths, amc_paths, sample_nums, modification)
        
        return data


    def prepare_inference(self, rotation_type, num_frames, model_path, tuning_length, num_tuning_cycles, at_loss_function, at_loss_parameters, at_learning_rate_binding, at_learning_rate_state, at_momentum_binding):
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
        self.BAPTAT.set_comparison_values(
            torch.Tensor(np.identity(self.num_dimensions)))
        
        info_string = ''
        info_string += f' - modification of body rotation: \t{rotation_type} by {self.new_rotation}\n\n'
        info_string = self.construct_info_string(info_string, at_loss_parameters)
        info_string += f' - learning rate: \t\t{self.BAPTAT.at_learning_rate}\n'
        info_string += f' - momentum: \t\t\t{self.BAPTAT.r_momentum}\n'
        
        self.write_to_file(info_string, self.result_path+'parameter_information.txt')
        print('Ready to run AT inference for rotation task! \nInitialized parameters with: \n' + info_string)
        

    def evaluate(self, 
        observations, 
        final_predictions, 
        final_rotation_values, 
        final_rotation_matrix, 
        feature_names, 
        rotation):

        results, figures = super().evaluate(self.BAPTAT, observations, final_predictions)
        ## Save figures
        figures += [self.BAPTAT.evaluator.plot_at_losses(results[2], 'History of rotation matrix loss (MSE)')]
        figures += [self.BAPTAT.evaluator.plot_at_losses(results[3], 'History of rotation angles loss (AL)')]


        names = [
            'prediction_errors', 'at_loss_history', 'rotmat_loss_history', 'rotang_loss_history', 
            'final_rotation_values', 'final_rotation_matrix']

        self.save_figures(figures, names)
        self.save_results_to_csv(results, names)

        i = len(results)
        torch.save(final_rotation_values, self.result_path + names[i] + '.pt')
        i += 1
        torch.save(final_rotation_matrix, self.result_path + names[i] + '.pt')

        return results



    def run(self, 
        experiment_dir,
        rotation_type,
        modified,
        sample_nums, 
        model_path, 
        tuning_length, 
        num_tuning_cycles, 
        at_loss_function,
        loss_parameters,
        at_learning_rate_rotation, 
        at_learning_rate_state, 
        at_momentum_rotation):

        print('*************************************************************************************')

        experiment_results = []
        res_path = self.trial_path + experiment_dir
        if experiment_dir != "":
            os.mkdir(res_path)
            print('Created directory: '+ res_path)


        data = self.load_data(
            modify=modified, 
            rotation_type=rotation_type, 
            sample_nums=sample_nums)


        for d in data:
            name = d[0]
            observations = d[1]
            feat_names = d[2]
            num_frames = observations.size()[0]
            self.BAPTAT.set_data_parameters_(num_frames, self.num_observations, self.num_dimensions)

            self.result_path = res_path+name+'/'
            os.mkdir(self.result_path)

            new_rotation = self.new_rotation
            if new_rotation is not None and '_' not in name:
                new_rotation = None
            
            
            self.prepare_inference(
                rotation_type,
                num_frames, 
                model_path, 
                tuning_length, 
                num_tuning_cycles, 
                at_loss_function, 
                loss_parameters,
                at_learning_rate_rotation, 
                at_learning_rate_state, 
                at_momentum_rotation)

            at_final_predictions, final_rotation_values, final_rotation_matrix = self.BAPTAT.run_inference(observations, new_rotation)

            # rerotate observations to compare with final predictions 
            if new_rotation is not None:
                observations = self.order(observations, self.reorder)

            res = self.evaluate(
                observations, 
                at_final_predictions, 
                final_rotation_values, 
                final_rotation_matrix, 
                feat_names, 
                new_rotation)
            print('Evaluated current run.')

            super().terminate()
            print('Terminated current run.')

            experiment_results += [[name, res, final_rotation_values, final_rotation_matrix]]

        return experiment_results
      