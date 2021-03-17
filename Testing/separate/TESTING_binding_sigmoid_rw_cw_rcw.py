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


class TEST_sigVSrwVScwVSrcw(TEST_BINDING): 

    def __init__(self, num_features, num_observations, num_dimensions):
        experiment_name = "binding_sigmoid_rw_cw_rcw"
        super().__init__(num_features, num_observations, num_dimensions, experiment_name)

        print('Initialized experiment.')


    def perform_experiment(self, sample_nums, scalers): 

        ## General parameters
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


        for scaler in scalers:
            self.BAPTAT.set_scale_mode(scaler)

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
                at_momentum_binding
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

    # sample_nums = [1000, 250, 300]
    # sample_nums = [250,250,250]
    # sample_nums = [100,100,100]
    # sample_nums = [20,20,20]
    # sample_nums = [50,50,50]
    # sample_nums = [15,15,15]
    sample_nums = [12,12,12]

    scalers = ['sigmoid', 'rwSM', 'cwSM', 'rcwSM']

    test.perform_experiment(sample_nums, scalers)

    

if __name__ == "__main__":
    main()