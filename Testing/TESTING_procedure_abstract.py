import pandas as pd   
import matplotlib.pyplot as plt
import abc 
from abc import ABC, abstractmethod

import torch 
from datetime import datetime
import os
import sys
sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from Data_Compiler.data_preparation import Preprocessor

class TEST_PROCEDURE(ABC): 

    def __init__(self, num_features, num_observations, num_dimensions): 
        self.num_features = num_features
        self.num_observations = num_observations
        self.num_dimensions = num_dimensions

        self.preprocessor = Preprocessor(self.num_observations, self.num_features, self.num_dimensions)

        print('Initialized test procedure.')


    def load_data():
        pass

        
    def create_trial_directory(self, path): 
        if not os.path.isdir(path):
            os.mkdir(path)
        
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%Y_%b_%d-%H_%M_%S")
        self.result_path = path+timestamp+'/'
        os.mkdir(self.result_path)

    
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

    
    def load_data_all(self, asf_paths, amc_paths, sample_nums, modification):         
        data = []

        for i in range(len(sample_nums)): 
            optimal_data= self.load_data_original(
                asf_paths[i], 
                amc_paths[i], 
                sample_nums[i])
            data += [optimal_data]
            

            if modification is not None:
                modified_data = self.load_data_modified(
                    asf_paths[i], 
                    amc_paths[i], 
                    sample_nums[i], 
                    modification)
                data += [modified_data]

        return data


    def load_data_original(self, asf_path, amc_path, num_samples): 
        observations, joint_names = self.preprocessor.get_AT_data(asf_path, amc_path, num_samples)
        name, _ = asf_path.split('.')
        _ , name = name.split('/')
        return (name, observations, joint_names)
        


    def load_data_modified(self, asf_path, amc_path, num_samples, modification): 
        (name, data, joint_names) = self.load_data_original(asf_path, amc_path, num_samples)
        original_shape = data.shape
        # TODO modify
        for mode, modify in modification: 
            if mode == 'rebind': 
                data = self.order(data, modify)
                print("Rebinded", name)
                name += "_rebinded"

            elif mode == 'qrotate':
                data = data.view(num_samples*self.num_observations, self.num_dimensions)
                data = self.PERSP_TAKER.qrotate(data, modify)   
                data = data.view(original_shape) 

                print("Q-Rotated", name)
                name += "_qrotated"

            elif mode == 'eulrotate':
                data = data.view(num_samples*self.num_observations, self.num_dimensions)
                rotmat = self.PERSP_TAKER.compute_rotation_matrix_(modify[0], modify[1], modify[2])
                data = self.PERSP_TAKER.rotate(data, rotmat)   
                data = data.view(original_shape)  
                
                print("Euler-Rotated", name)
                name += "_eulrotated"

            elif mode == 'translate':
                data = self.PERSP_TAKER.translate(data, modify)
                print("Translated", name)
                name += "_translated"
            else: 
                print('Unknown modification ', mode, ' for ', name, ' was skipped.')  


        return (name, data, joint_names)


    def order(self, data, order): 
        return data.gather(1, order.unsqueeze(1).expand(data.shape))


    def prepare_inference(self, 
        baptat, 
        num_frames, 
        model_path, 
        tuning_length, 
        num_tuning_cycles, 
        at_loss_function, 
        at_learning_rate, 
        at_learning_rate_state, 
        at_momentum):

        baptat.init_model_(model_path)
        baptat.set_tuning_parameters_(
            tuning_length, 
            num_tuning_cycles, 
            at_loss_function, 
            at_learning_rate, 
            at_learning_rate_state, 
            at_momentum
        )
        baptat.init_inference_tools()


    def construct_info_string(self, info_string, loss_parameters):
        info_string += f' - number of observations: \t{self.BAPTAT.num_input_features}\n'
        info_string += f' - number of features: \t\t{self.BAPTAT.num_input_features}\n'
        info_string += f' - number of dimensions: \t{self.BAPTAT.num_input_dimensions}\n'
        info_string += f' - number of tuning cycles: \t{self.BAPTAT.tuning_cycles}\n'
        info_string += f' - size of tuning horizon: \t{self.BAPTAT.tuning_length}\n'
        info_string += f' - loss function: \t\t{self.BAPTAT.at_loss}\n'
        for name, value in loss_parameters:
            info_string += f'\t> {name}: \t{value}\n'
        info_string += f' - model: \t\t\t{self.BAPTAT.core_model}\n'
        info_string += f' - learning rate (state): \t{self.BAPTAT.at_learning_rate_state}\n'

        return info_string



    def evaluate(self, baptat, observations, final_predictions):
        results = baptat.get_result_history(observations, final_predictions)
        figures = []
        figures += [baptat.evaluator.plot_prediction_errors(results[0])]
        figures += [baptat.evaluator.plot_at_losses(results[1], 'History of overall losses during active tuning')]
        
        return results, figures


    def save_tesor_to_csv(self, tensor, path): 
        torch.save(tensor, path)


    def save_figures(self, figures, names):
        for i in range(len(figures)): 
            fig = figures[i]
            fig.savefig(self.result_path + names[i] + '.png', bbox_inches='tight', dpi=150)

    
    def write_to_file(self, string, path):
        out = open(path, "wt")
        out.write(string)
        out.close()


    def save_results_to_csv(self, results, names): 
        for i in range(len(results)):
            df = pd.DataFrame(results[i])  
            df.to_csv(self.result_path + names[i] + '.csv')


    def save_results_to_pt(self, results, names): 
        for i in range(len(results)):
            torch.save(results[i], self.result_path + names[i] + '.pt')
  

    def run(): 
        pass


    def terminate(self):
        plt.close('all')
    
