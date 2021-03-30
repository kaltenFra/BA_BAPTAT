import numpy as np
import torch 
from torch import nn
from datetime import datetime
import os
import pandas as pd

import sys

sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from interfaces.general_interface import TESTER
from Testing.TESTING_statistical_evaluation_abstract import TEST_STATISTICS


class COMP_ROTATION_VALUES_GEST(TESTER): 

    def __init__(self, num_features, num_observations, num_dimensions):
        experiment_name = f"compare_rotations_dim{num_dimensions}"
        super().__init__(num_features, num_observations, num_dimensions, experiment_name)
        self.stats = TEST_STATISTICS(self.num_features, self.num_observations, self.num_dimensions)
        print('Initialized experiment.')


    def perform_experiment(self, sample_nums, rotations, rotation_type): 

        ## General parameters
        # possible loss functions
        mse = nn.MSELoss()
        l1Loss = nn.L1Loss()
        smL1Loss = nn.SmoothL1Loss(reduction='sum')
        l2Loss = lambda x,y: self.mse(x, y) * (self.num_dimensions * self.num_observations)


        # possible models to use 
        if self.num_dimensions == 7:
            model_path = 'BA_BAPTAT/CoreLSTM/models/ADAM/LSTM_24_gest.pt'
            # self.model_path = 'CoreLSTM/models/ADAM/LSTM_24_gest.pt'
        elif self.num_dimensions == 6:
            model_path = 'BA_BAPTAT/CoreLSTM/models/ADAM/LSTM_25_vel.pt'
            # self.model_path = 'CoreLSTM/models/ADAM/LSTM_25_vel.pt'
        elif self.num_dimensions == 3: 
            model_path = 'BA_BAPTAT/CoreLSTM/models/ADAM/LSTM_26_pos.pt'
            # model_path = 'CoreLSTM/models/ADAM/LSTM_26_pos.pt'
        else: 
            print('ERROR: Unvalid number of dimensions!\nPlease use 3, 6, or 7.')
            exit()

        ## Experiment parameters
        # NOTE: set manually, for all parameters that are not tested

        tuning_length = 20
        num_tuning_cycles = 3
        # at_loss_function = mse
        at_loss_function = nn.SmoothL1Loss(reduction='sum', beta=0.0001)
        loss_parameters = [('beta', 0.0001), ('reduction', 'sum')]


        at_learning_rate_binding = 0.1
        at_learning_rate_rotation =  0.1
        at_learning_rate_translation = 1
        at_learning_rate_state = 0.0

        at_momentum_binding = 0.9
        at_momentum_rotation = 0.8
        at_momentum_translation = 0.3

        grad_calc_binding = 'weightedInTunHor'
        grad_calc_rotation = 'meanOfTunHor'
        grad_calc_translation = 'meanOfTunHor'
        grad_calculations = [grad_calc_binding, grad_calc_rotation, grad_calc_translation]
        
        grad_bias_binding = 1.4
        grad_bias_rotation = 1.1
        grad_bias_translation = 1.5 
        grad_biases = [grad_bias_binding, grad_bias_rotation, grad_bias_translation]

        nxm_bool = False
        index_additional_features = [6]
        initial_value_dummie_line = 0.1
        nxm_enhancement = 'square'
        nxm_outcast_line_scaler = 0.1

        initial_axis_angle = 0

        experiment_results = []

        ## experiment performed for all values of the tested parameter
        changed_parameter = 'rotations'
        parameter_values = rotations

        for rot in rotations: 
            # set value of tested parameter
            modification = [
                ('rotate', 'set', (rotation_type, rot))
            ]
            
            self.BAPTAT.set_rotation_type(rotation_type)
            print(f'New rotation: {rot}')

            # set BAPTAT parameters
            self.BAPTAT.set_weighted_gradient_biases(grad_biases)  
            self.BAPTAT.set_init_axis_angle(initial_axis_angle)


            # run experiment
            sample_names, result_names, results = super().run(
                changed_parameter+"_"+str(rot)+"/",
                modification,
                sample_nums, 
                model_path, 
                tuning_length, 
                num_tuning_cycles, 
                at_loss_function,
                loss_parameters,
                [at_learning_rate_binding, at_learning_rate_rotation, at_learning_rate_translation], 
                at_learning_rate_state, 
                [at_momentum_binding, at_momentum_rotation, at_momentum_translation],
                grad_calculations
            )

            experiment_results += [results]

        
        ## statistics and comparison plots 

        dfs = self.stats.load_csvresults_to_dataframe(
            self.prefix_res_path, 
            changed_parameter, 
            parameter_values, 
            sample_names, 
            result_names
            )


        self.stats.plot_histories(
            dfs, 
            self.prefix_res_path, 
            changed_parameter, 
            result_names, 
            result_names
        )

        self.stats.plot_value_comparisons(
            dfs, 
            self.prefix_res_path, 
            changed_parameter, 
            result_names, 
            result_names
        )



        print("Terminated experiment.")


            
       
def main(): 
    # set the following parameters
    num_observations = 15
    num_input_features = 15
    num_dimensions = 6

    test = COMP_ROTATION_VALUES_GEST(
        num_input_features, 
        num_observations, 
        num_dimensions) 


    # sample_nums = [990, 550]
    sample_nums = [300, 300]
    # sample_nums = [22, 22]


    # rotation_type = 'eulrotate'
    rotation_type = 'qrotate'

    rotations = [
        # torch.Tensor([ 1.0, 0.0, 0.0, 0.0 ]) # eul: 0, 0, 0 | axang: 0
        # ,torch.Tensor([ 0.9238795, 0.2209424, 0.2209424, 0.2209424 ]) # eul: x: 32.1545463, y: 18.0964303, z: 32.1545463 | axang: 45
        # ,torch.Tensor([ 0.7071068 , 0.4082483, 0.4082483, 0.4082483 ]) # eul: x: 69.8960907, y: 14.1237448, z: 69.8960907 | axang: 90
        # ,torch.Tensor([ 0.3826834, 0.5334021, 0.5334021, 0.5334021 ]) # eul: 0, 0, 0 | axang: 135
        # ,torch.Tensor([ 0.0, 0.5773503, 0.5773503, 0.5773503 ]) # eul: 0, 0, 0 | axang: 180
        # ,torch.Tensor([ -0.3826834, 0.5334021, 0.5334021, 0.5334021 ]) # eul: 0, 0, 0 | axang: 225
        # ,torch.Tensor([ -0.7071068 , 0.4082483, 0.4082483, 0.4082483 ]) # eul: 0, 0, 0 | axang: 270
        # ,torch.Tensor([ -0.9238795, 0.2209424, 0.2209424, 0.2209424 ]) # eul: 0, 0, 0 | axang: 315
        # ,torch.Tensor([ -1.0, 0.0, 0.0, 0.0 ]) # eul: 0, 0, 0 | axang: 360

        #### angles of robust inference paper
        torch.Tensor([ 0.7860946, 0.0570724, 0.5421742, 0.291282 ]) # eul: 45, 55, 65 | axang: 76.3559468
        ,torch.Tensor([ 0.8751774 , 0.0799913, 0.4356247, 0.1946717 ]) # eul: 27, 47, 37 | axang: 57.8680499
        ,torch.Tensor([ 0.9047343 , -0.0434052, 0.2160705, 0.3645344 ]) # eul: 5, 25, 45 | axang: 50.4249533
        ,torch.Tensor([ 0.7556698 , 0.4048287, 0.5117495, -0.0564742 ]) # eul: 75, 55, 35 | axang: 81.8321326
        ,torch.Tensor([ 0.7445376 , 0.258584, 0.6151776, 0.0188295 ]) # eul: 75, 65, 55 | axang: 83.7612
        ,torch.Tensor([ 0.7656519 , -0.1862782, 0.5430848, 0.290063 ]) # eul: 5, 70, 45 | axang: 80.0699577
        ,torch.Tensor([ 0.7277808 , -0.0055396, 0.643789, 0.2363052 ]) # eul: 60, 70, 80 | axang: 86.5986554
        ,torch.Tensor([ 0.7277808 , 0.2363052, 0.643789, -0.0055396 ]) # eul: 80, 70, 60 | axang: 86.5986554
        ,torch.Tensor([ 0.7333283 , -0.0751105, 0.5503156, 0.3920978 ]) # eul: 40, 60, 80 | axang: 85.6676966
        ,torch.Tensor([ 0.8346984, 0.1954527, 0.5089286, 0.0779006 ]) # eul: 45, 55, 35 | axang: 66.8311099
    ]

    test.perform_experiment(sample_nums, rotations, rotation_type)

    

if __name__ == "__main__":
    main()