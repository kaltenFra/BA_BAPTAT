import pandas as pd   
import abc 
from abc import ABC, abstractmethod

import torch 

import sys
sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from Data_Compiler.data_preparation import Preprocessor

class TEST_PROCEDURE(ABC): 

    def __init__(self, num_features, num_observations, num_dimensions): 
        self.num_features = num_features
        self.num_observations = num_observations
        self.num_dimensions = num_dimensions

        self.preprocessor = Preprocessor(self.num_features, self.num_dimensions)

        print('Initialized test procedure.')

    
    def load_data_all(self, asf_paths, amc_paths, sample_nums, train_modification): 
        data = []

        optimal_train_data = self.load_data_original(
            asf_paths[0], 
            amc_paths[0], 
            sample_nums[0])
        data += [optimal_train_data]

        if train_modification is not None:
            modified_train_data = self.load_data_modified(
                asf_paths[0], 
                amc_paths[0], 
                sample_nums[0], 
                train_modification)
            data += [modified_train_data]

        for i in range(len(amc_paths)-1): 
            test_data= self.load_data_original(
                asf_paths[i+1], 
                amc_paths[i+1], 
                sample_nums[i+1])
            data += [test_data]

        return data


    def load_data_original(self, asf_path, amc_path, num_samples): 
        observations, joint_names = self.preprocessor.get_AT_data(asf_path, amc_path, num_samples)
        name, _ = asf_path.split('.')
        _ , name = name.split('/')
        return (name, observations, joint_names)
        


    def load_data_modified(self, asf_path, amc_path, num_samples, modification): 
        (name, optim_data, joint_names) = self.load_data_train_optimal(asf_path, amc_path, num_samples)
        # TODO modify
        mod_data = optim_data

        return (name, mod_data, joint_names)


    def prepare_inference(self, baptat, num_frames, model_path, tuning_length, num_tuning_cycles, at_loss_function, at_learning_rate_binding, at_learning_rate_state, at_momentum_binding):
        baptat.set_data_parameters_(num_frames, self.num_features, self.num_dimensions)
        baptat.init_model_(model_path)
        baptat.set_tuning_parameters_(
            tuning_length, 
            num_tuning_cycles, 
            at_loss_function, 
            at_learning_rate_binding, 
            at_learning_rate_state, 
            at_momentum_binding
        )
        baptat.init_inference_tools()


    def evaluate(self, baptat, observations, final_predictions):
        results = baptat.get_result_history(observations, final_predictions)
        figures = []
        figures += [baptat.evaluator.plot_prediction_errors(results[0])]
        figures += [baptat.evaluator.plot_at_losses(results[1], 'History of overall losses during active tuning')]
        
        return results, figures


    def save_tesor_to_csv(self, tensor, path): 
        torch.save(tensor, path)


    def save_figure(self, figure, path):
        ...

    
    def write_to_file(self, string, path):
        out = open(path, "wt")
        out.write(string)
        out.close()


    def save_array_to_csv(self, array_like_rows, col_names, path): 
        # creating and saving dataframe
        df = pd.DataFrame(array_like_rows, columns = col_names)  
        df.to_csv('GFG.csv') 
  
     

    def run(): 
        pass



    
