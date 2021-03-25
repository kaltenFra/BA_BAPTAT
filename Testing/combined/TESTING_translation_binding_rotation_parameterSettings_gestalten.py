import numpy as np
import torch 
from torch import nn
from datetime import datetime
import os
import pandas as pd

import sys

sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from interfaces.combined_interface_gestalten import TEST_COMBINATIONS_GESTALTEN


class TEST_COMBI_ALL_PARAMS(TEST_COMBINATIONS_GESTALTEN): 

    def __init__(self, num_features, num_observations, num_dimensions):
        experiment_name = "combination_t_b_r_gest_parameter_settings"
        super().__init__(num_features, num_observations, num_dimensions, experiment_name)

        print('Initialized experiment.')


    def perform_experiment(self, sample_nums, changed_parameter, parameter_values): 

        ## General parameters
        # possible loss functions
        mse = nn.MSELoss()
        l1Loss = nn.L1Loss()
        smL1Loss = nn.SmoothL1Loss(reduction='sum')
        l2Loss = lambda x,y: self.mse(x, y) * (self.num_dimensions * self.num_observations)

        # set manually
        modification = [
            ('bind', None, None),
            ('rotate', None, 'qrotate'), 
            ('translate', None, range(5))
            # TODO: FIRST change part in Abstract
            # ('bind', 'det', None),
            # ('rotate', 'det', 'qrotate'), 
            # ('translate', 'det', range(5))
        ]

        model_path = 'CoreLSTM/models/LSTM_73_gestalten.pt'
        # model_path = 'CoreLSTM/models/LSTM_69_gestalten.pt'
        tuning_length = 20
        num_tuning_cycles = 3
        at_loss_function = nn.SmoothL1Loss(reduction='mean', beta=0.1)
        loss_parameters = [('beta', 0.1), ('reduction', 'mean')]

        rotation_type = 'qrotate'

        at_learning_rate_binding = 1
        at_learning_rate_rotation = 0.1
        at_learning_rate_translation = 1
        at_learning_rate_state = 0.1

        at_momentum_binding = 0.5
        at_momentum_rotation = 0.8
        at_momentum_translation = 0.3

        grad_calc_binding = 'weightedInTunHor'
        grad_calc_rotation = 'weightedInTunHor'
        grad_calc_translation = 'meanOfTunHor'
        grad_calculations = [grad_calc_binding, grad_calc_rotation, grad_calc_translation]
        
        grad_bias_binding = 1.5
        grad_bias_rotation = 1.2 
        grad_bias_translation = 1.5 
        grad_biases = [grad_bias_binding, grad_bias_rotation, grad_bias_translation]


        for val in parameter_values: 
            if changed_parameter == 'model_path': 
                model_path = val
            elif changed_parameter == 'modification': 
                modification = val
            elif changed_parameter == 'rotation_type': 
                rotation_type = val
            elif changed_parameter == 'tuning_length': 
                tuning_length = val
            elif changed_parameter == 'num_tuning_cycles': 
                num_tuning_cycles = val
            elif changed_parameter == 'at_loss_function': 
                at_loss_function = val
            elif changed_parameter == 'loss_parameters': 
                [(_, beta_val), (_, reduction_val)] = loss_parameters
                at_loss_function = nn.SmoothL1Loss(reduction=reduction_val, beta=beta_val)
            elif changed_parameter == 'at_learning_rate_binding': 
                at_learning_rate_binding = val
            elif changed_parameter == 'at_learning_rate_rotation': 
                at_learning_rate_rotation = val
            elif changed_parameter == 'at_learning_rate_translation': 
                at_learning_rate_translation = val
            elif changed_parameter == 'at_learning_rate_state': 
                at_learning_rate_state = val
            elif changed_parameter == 'at_momentum_binding': 
                at_momentum_binding = val  
            elif changed_parameter == 'at_momentum_rotation': 
                at_momentum_rotation = val 
            elif changed_parameter == 'at_momentum_translation': 
                at_momentum_translation = val 
            elif changed_parameter == 'grad_calculations': 
                grad_calculations = val  
            elif changed_parameter == 'grad_biases': 
                grad_biases = val  
            else: 
                print('Unknown parameter!')
                break       

            print(f'New value for {changed_parameter}: {val}')

            self.BAPTAT.set_weighted_gradient_biases(grad_biases)  
            self.BAPTAT.set_rotation_type(rotation_type)          

            results = super().run(
                changed_parameter+"_"+str(val)+"/",
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



        print("Terminated experiment.")
            
      
def main(): 
    num_observations = 15
    num_input_features = 15
    num_dimensions = 7
    test = TEST_COMBI_ALL_PARAMS(
        num_observations, 
        num_input_features, 
        num_dimensions) 

    
    # sample_nums = [1000, 250, 300]
    # sample_nums = [250,250,250]
    # sample_nums = [100,100,100]
    # sample_nums = [20,20,20]
    # sample_nums = [50,50,50]
    # sample_nums = [15,15,15]
    # sample_nums = [12,12,12]

    sample_nums = [300]

    tested_parameter = 'num_tuning_cycles'
    # parameter_values = [1, 2, 3]
    parameter_values = [3]

    # tested_parameter = 'at_loss_function'
    # parameter_values = [nn.SmoothL1Loss(reduction='sum', beta=0.8), nn.MSELoss()]

    # tested_parameter = 'loss_parameters'
    # parameter_values = [
    #     [('beta', 0.4),('reduction', 'mean')], 
    #     [('beta', 0.6),('reduction', 'mean')], 
    #     [('beta', 0.8),('reduction', 'mean')], 
    #     [('beta', 1.0),('reduction', 'mean')]
    #     [('beta', 1.2),('reduction', 'mean')], 
    # ]

    # tested_parameter = 'at_learning_rate_binding'
    # parameter_values = [1, 0.1, 0.01]


    # tested_parameter = 'at_learning_rate_translation'
    # parameter_values = [1, 0.1, 0.01]




    test.perform_experiment(sample_nums, tested_parameter, parameter_values)

    

if __name__ == "__main__":
    main()