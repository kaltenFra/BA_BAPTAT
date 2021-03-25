import numpy as np
import torch 
from torch import nn
from datetime import datetime
import os
import pandas as pd

import sys

sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from interfaces.binding_interface import TEST_BINDING


class TEST_sigVSrwVScwVSrcw(TEST_BINDING): 

    def __init__(self, num_features, num_observations, num_dimensions):
        experiment_name = "binding_sigmoid_rw_cw_rcw"
        super().__init__(num_features, num_observations, num_dimensions, experiment_name)

        print('Initialized experiment.')


    def perform_experiment(self, sample_nums, scalers): 

        ## General parameters
        mse = nn.MSELoss()
        l1Loss = nn.L1Loss()
        smL1Loss = nn.SmoothL1Loss(reduction='sum')
        euklLoss = lambda x,y: mse(x, y) * (self.num_dimensions * self.num_observations)
        # set manually
        modified = 'det'
        model_path = 'CoreLSTM/models/LSTM_46_cell.pt'
        tuning_length = 10
        num_tuning_cycles = 3
        at_loss_function = nn.SmoothL1Loss(reduction='sum', beta=0.3)
        # at_loss_function = euklLoss
        loss_parameters = [('beta', 0.3), ('reduction', 'sum')]
        at_learning_rate_binding = 1
        at_learning_rate_state = 0.1
        at_momentum_binding = 0.9

        grad_calc = 'weightedInTunHor'
        grad_bias = 1.5

        for scaler in scalers:
            self.BAPTAT.set_scale_mode(scaler)
            self.BAPTAT.set_weighted_gradient_bias(grad_bias)

            results = super().run(
                scaler+'/',
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
    test = TEST_sigVSrwVScwVSrcw(
        num_observations, 
        num_input_features, 
        num_dimensions)    

    # sample_nums = [500, 500, 500, 500, 500]
    sample_nums = [1000, 550, 450, 300, 250]
    # sample_nums = [250,250,250]
    # sample_nums = [100,100,100]
    # sample_nums = [20,20,20]
    # sample_nums = [50,50,50]
    # sample_nums = [15,15,15]
    # sample_nums = [12,12,12]
    # sample_nums = [300]
    

    scalers = ['sigmoid', 'rwSM', 'cwSM', 'rcwSM']
    # scalers = ['rcwSM']

    test.perform_experiment(sample_nums, scalers)

    

if __name__ == "__main__":
    main()