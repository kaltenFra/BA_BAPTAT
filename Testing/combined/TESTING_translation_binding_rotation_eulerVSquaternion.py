import numpy as np
import torch 
from torch import nn
from datetime import datetime
import os
import pandas as pd

import sys

sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from interfaces.combined_interface import TEST_COMBINATIONS


class TEST_COMBI_ALL_EULERvsQUATERNION(TEST_COMBINATIONS): 

    def __init__(self, num_features, num_observations, num_dimensions):
        experiment_name = "combination_all_eulerVSquaternion"
        super().__init__(num_features, num_observations, num_dimensions, experiment_name)

        print('Initialized experiment.')


    def perform_experiment(self, sample_nums, rotation_types): 

        ## General parameters
        # possible loss functions
        mse = nn.MSELoss()
        l1Loss = nn.L1Loss()
        smL1Loss = nn.SmoothL1Loss(reduction='sum')
        l2Loss = lambda x,y: self.mse(x, y) * (self.num_dimensions * self.num_observations)

        # set manually
        modification = [
            ('bind', 'det', None), 
            ('rotate', 'det', 'qrotate'), 
            ('translate', 'det', range(5))
        ]
        # rotation_type = 'eulrotate'
        rotation_type = 'qrotate'
        model_path = 'CoreLSTM/models/LSTM_46_cell.pt'
        tuning_length = 10
        num_tuning_cycles = 3
        at_loss_function = nn.SmoothL1Loss(reduction='mean', beta=1)
        loss_parameters = [('beta', 1), ('reduction', 'mean')]

        at_learning_rate_binding = 1
        at_learning_rate_rotation = 1
        at_learning_rate_translation = 1
        at_learning_rate_state = 0.0

        at_momentum_binding = 0.0
        at_momentum_rotation = 0.0
        at_momentum_translation = 0.0

        grad_calc_binding = 'weightedInTunHor'
        grad_calc_rotation = 'meanOfTunHor'
        grad_calc_translation = 'meanOfTunHor'
        grad_calculations = [grad_calc_binding, grad_calc_rotation, grad_calc_translation]
        
        grad_bias_binding = 1.5 
        grad_bias_rotation = 1.5 
        grad_bias_translation = 1.5 
        grad_biases = [grad_bias_binding, grad_bias_rotation, grad_bias_translation]


        for type in rotation_types:

            self.BAPTAT.set_weighted_gradient_biases(grad_biases)  
            self.BAPTAT.set_rotation_type(type)          

            results = super().run(
                'rotationType_'+type+"/",
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
    num_dimensions = 3
    test = TEST_COMBI_ALL_EULERvsQUATERNION(
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
    sample_nums = [30]

    rotation_types = ['eulrotate', 'qrotate']
    # rotation_types = ['eulrotate']
    # rotation_types = ['qrotate']
        

    test.perform_experiment(sample_nums, rotation_types)

    

if __name__ == "__main__":
    main()