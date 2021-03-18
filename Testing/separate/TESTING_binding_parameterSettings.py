import numpy as np
import torch 
from torch import nn
from datetime import datetime
import os
import pandas as pd

import sys

sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from interfaces.binding_interface import TEST_BINDING
from BAPTAT_3_binding_class import SEP_BINDING


class TEST_BINDING_PARAMS(TEST_BINDING): 

    def __init__(self, num_features, num_observations, num_dimensions):
        experiment_name = "binding_parameter_settings"
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
        tuning_length = 10
        num_tuning_cycles = 1
        at_loss_function = nn.SmoothL1Loss(reduction='sum', beta=0.8)
        loss_parameters = [('beta', 0.8), ('reduction', 'sum')]
        at_learning_rate_binding = 1
        at_learning_rate_state = 0.0
        at_momentum_binding = 0.1

        grad_calc = 'meanOfTunHor'
        grad_bias = 1.5 

        for val in parameter_values: 
            if changed_parameter == 'model_path': 
                model_path = val
            elif changed_parameter == 'tuning_length': 
                tuning_length = val
            elif changed_parameter == 'num_tuning_cycles': 
                num_tuning_cycles = val
            elif changed_parameter == 'at_loss_function': 
                at_loss_function = val
            elif changed_parameter == 'loss_parameters': 
                loss_parameters = val
            elif changed_parameter == 'at_learning_rate_binding': 
                at_learning_rate_binding = val
            elif changed_parameter == 'at_learning_rate_state': 
                at_learning_rate_state = val
            elif changed_parameter == 'at_momentum_binding': 
                at_momentum_binding = val  
            elif changed_parameter == 'grad_calc': 
                grad_calc = val  
            elif changed_parameter == 'grad_bias': 
                grad_bias = val  
            else: 
                print('Unknown parameter!')
                break       

            print(f'New value for {changed_parameter}: {val}')

            self.BAPTAT.set_weighted_gradient_bias(grad_bias)

            results = super().run(
                changed_parameter+"_"+str(val)+"/",
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
                grad_calc
            )


        print("Terminated experiment.")
            
      
def main(): 
    num_observations = 15
    num_input_features = 15
    num_dimensions = 3
    test = TEST_BINDING_PARAMS(
        num_observations, 
        num_input_features, 
        num_dimensions) 

    
    # sample_nums = [1000, 250, 300]
    # sample_nums = [250,250,250]
    # sample_nums = [100,100,100]
    # sample_nums = [20,20,20]
    # sample_nums = [50,50,50]
    # sample_nums = [15,15,15]
    sample_nums = [12,12,12]
    # sample_nums = [12]

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