import numpy as np
import torch 
from torch import nn
from datetime import datetime
import os
import pandas as pd

import sys

sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from interfaces.rotation_interface import TEST_ROTATION


class TEST_ROTATION_PARAMS(TEST_ROTATION): 

    def __init__(self, num_features, num_observations, num_dimensions):
        experiment_name = "rotation_parameter_settings"
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
        modified = 'det'
        model_path = 'CoreLSTM/models/LSTM_46_cell.pt'
        # rotation_type = 'eulrotate'
        rotation_type = 'qrotate'

        tuning_length = 10
        num_tuning_cycles = 3
        at_loss_function = mse
        # at_loss_function = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        # loss_parameters = [('beta', 1.0), ('reduction', 'mean')]
        loss_parameters = []
        at_learning_rate_rotation = 1
        at_learning_rate_state = 0.0 
        at_momentum_rotation = 0.1

        grad_calc = 'meanOfTunHor'
        grad_bias = 1.5 


        for val in parameter_values: 
            if changed_parameter == 'model_path': 
                model_path = val
            elif changed_parameter == 'modified': 
                modified = val
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
            elif changed_parameter == 'at_learning_rate_rotation': 
                at_learning_rate_rotation = val
            elif changed_parameter == 'at_learning_rate_state': 
                at_learning_rate_state = val
            elif changed_parameter == 'at_momentum_rotation': 
                at_momentum_rotation = val  
            elif changed_parameter == 'grad_calc': 
                grad_calc = val  
            elif changed_parameter == 'grad_bias': 
                grad_bias = val  
            else: 
                print('Unknown parameter!')
                break       

            print(f'New value for {changed_parameter}: {val}')


            self.BAPTAT.set_weighted_gradient_bias(grad_bias)
            self.BAPTAT.set_rotation_type(rotation_type)


            results = super().run(
                changed_parameter+"_"+str(val)+"/",
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
                at_momentum_rotation,
                grad_calc
            )

        print("Terminated experiment.")
            
      
        

def main(): 
    num_observations = 15
    num_input_features = 15
    num_dimensions = 3
    test = TEST_ROTATION_PARAMS(
        num_observations, 
        num_input_features, 
        num_dimensions)  

    ## Number of samples per motion capture dataset. 
    #   -> Size of list automatically specifies number of used datasets. (fixed order)
    # sample_nums = [1000, 250, 300]
    # sample_nums = [250,250,250]
    # sample_nums = [100,100,100]
    # sample_nums = [20,20,20]
    # sample_nums = [50,50,50]
    # sample_nums = [15,15,15]
    # sample_nums = [12,12,12]
    sample_nums = [20]

    tested_parameter = 'num_tuning_cycles'
    parameter_values = [1,3]

    # tested_parameter = 'at_loss_function'
    # parameter_values = [nn.SmoothL1Loss(reduction='sum', beta=0.8), nn.MSELoss()]

    # tested_parameter = 'loss_parameters'
    # parameter_values = [
    #     [('beta', 0.8),('reduction', 'sum')], 
    #     [('beta', 1.0),('reduction', 'sum')], 
    #     [('beta', 0.8),('reduction', 'mean')], 
    #     [('beta', 1.0),('reduction', 'mean')]
    # ]

    test.perform_experiment(sample_nums, tested_parameter, parameter_values)

    

if __name__ == "__main__":
    main()