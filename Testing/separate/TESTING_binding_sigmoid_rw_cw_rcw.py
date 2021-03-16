import numpy as np
import torch 
from torch import nn
from datetime import datetime
import os
import pandas as pd

import sys

sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from Testing.TESTING_procedure_abstract import TEST_PROCEDURE
from BAPTAT_3_binding_class import SEP_BINDING


class TEST_sigVSrwVScwVSrcw(TEST_PROCEDURE): 

    def __init__(self, num_features, num_observations, num_dimensions):
        super().__init__(num_features, num_observations, num_dimensions)
        self.BAPTAT = SEP_BINDING()

        experiment_path = "D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT/Grafics/SeparateBindingRuns/sigmoid_vs_rw_vs_cw_vs_rcw/"
        if not os.path.isdir(experiment_path):
            os.mkdir(experiment_path)
        
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%Y_%b_%d-%H_%M_%S")
        self.result_path = experiment_path+timestamp+'/'
        os.mkdir(self.result_path)

        print('Initialized test environment.')

    
    def get_data_paths(self):
        data_amc_path = []
        data_asf_path = []
        # data from LSTM training
        data_amc_path += ['Data_Compiler/S35T07.amc']
        data_asf_path += ['Data_Compiler/S35T07.asf']

        # inference test data
        data_amc_path += ['Data_Compiler/S07T02.amc']
        data_asf_path += ['Data_Compiler/S07T02.asf']

        data_amc_path += ['Data_Compiler/S08T02.amc']
        data_asf_path += ['Data_Compiler/S08T02.asf']

        return data_amc_path, data_asf_path



    def load_data(self, modify=None):
        amc_paths, asf_paths = self.get_data_paths()
        # sample_nums = [1000, 250, 300]
        sample_nums = [250,250,250]

        modification = None
        if modify == 'rand': 
            modification = ...
        elif modify == 'det': 
            modification = ...
        
        data = super().load_data_all(asf_paths, amc_paths, sample_nums, modification)
        
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
        self.BAPTAT.set_comparison_values(
            torch.Tensor(np.identity(15)),
            True)
        # print(self.BAPTAT.ideal_binding)
        
        info_string = ''
        info_string += f' - number of observations: \t{self.BAPTAT.num_input_features}\n'
        info_string += f' - number of features: \t\t{self.BAPTAT.num_input_features}\n'
        info_string += f' - number of dimensions: \t{self.BAPTAT.num_input_dimensions}\n'
        info_string += f' - number of tuning cycles: \t{self.BAPTAT.tuning_cycles}\n'
        info_string += f' - size of tuning horizon: \t{self.BAPTAT.tuning_length}\n'
        info_string += f' - loss function: \t\t{self.BAPTAT.at_loss}\n'
        for name, value in at_loss_parameters:
            info_string += f'\t> {name}: \t{value}\n'
        info_string += f' - learning rate: \t\t{self.BAPTAT.at_learning_rate}\n'
        info_string += f' - learning rate (state): \t{self.BAPTAT.at_learning_rate_state}\n'
        info_string += f' - momentum: \t\t\t{self.BAPTAT.bm_momentum}\n'
        info_string += f' - model: \t\t\t{self.BAPTAT.core_model}\n'

        super().write_to_file(info_string, self.result_path+'parameter_information.txt')
        print('Ready to run AT inference for binding task! \nInitialized parameters with: \n' + info_string)
        exit()

        
        

    def evaluate(self, observations, final_predictions, final_binding_matrix, final_binding_entries, feature_names):
        results, figures = super().evaluate(self.BAPTAT, observations, final_predictions)
        ## Save figures
        figures += [self.BAPTAT.evaluator.plot_at_losses(results[2], 'History of binding matrix loss (FBE)')]
        figures += [self.BAPTAT.evaluator.plot_at_losses(results[3], 'History of binding matrix determinante')]
        figures += [self.BAPTAT.evaluator.plot_binding_matrix(
            final_binding_matrix, 
            feature_names, 
            'Binding matrix showing relative contribution of observed feature to input feature'
        )]
        figures += [self.BAPTAT.evaluator.plot_binding_matrix(
            final_binding_entries, 
            feature_names, 
            'Binding matrix entries showing contribution of observed feature to input feature'
        )]


        names = ['prediction_errors', 'at_loss_history', 'fbe_history', 'determinante_history', 'final_binding_matirx', 'final_binding_neurons_activities']
        for i in range(len(figures)): 
            fig = figures[i]
            fig.savefig(self.result_path + names[i] + '.png', bbox_inches='tight', dpi=150)

        ## Save results
        for i in range(len(results)):
            df = pd.DataFrame(results[i])  
            df.to_csv(self.result_path + names[i] + '.csv')

        i += 1
        torch.save(final_binding_matrix, self.result_path + names[i] + '.pt')
        i += 1
        torch.save(final_binding_entries, self.result_path + names[i] + '.pt')


    def run(self):
        data = self.load_data()
        res_path = self.result_path

        model_path = 'CoreLSTM/models/LSTM_46_cell.pt'
        tuning_length = 10
        num_tuning_cycles = 3
        at_loss_function = nn.SmoothL1Loss(reduction='sum')
        loss_parameters = [('beta', 0.8), ('reduction', 'sum')]
        at_learning_rate_binding = 1
        at_learning_rate_state = 0.0 
        at_momentum_binding = 0.0
        for d in data:
            self.result_path = res_path+d[0]+'/'
            os.mkdir(self.result_path)


            observations = d[1]
            feat_names = d[2]
            num_frames = observations.size()[0]

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

            at_final_predictions, final_binding_matrix, final_binding_entries = self.BAPTAT.run_inference(observations)

            self.evaluate(observations, at_final_predictions, final_binding_matrix, final_binding_entries, feat_names)




def main(): 
    test = TEST_sigVSrwVScwVSrcw(15, 15, 3)    

    test.run()

    

if __name__ == "__main__":
    main()