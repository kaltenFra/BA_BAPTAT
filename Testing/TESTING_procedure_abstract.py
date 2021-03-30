import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import abc 
from abc import ABC, abstractmethod

import torch 
from datetime import datetime
import os
import sys
sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from Data_Compiler.data_preparation import Preprocessor
from Data_Compiler.skeleton_renderer import SKEL_RENDERER

class TEST_PROCEDURE(ABC): 

    def __init__(self, num_features, num_observations, num_dimensions): 
        self.num_features = num_features
        self.num_observations = num_observations
        self.num_dimensions = num_dimensions
        self.gestalten = False
        self.dir_mag_gest = False
        if num_observations==17: 
            self.data_distractor = True
        else:
            self.data_distractor = False

        self.preprocessor = Preprocessor(
            self.num_observations, 
            self.num_features, 
            self.num_dimensions, 
            self.data_distractor)
            
        # fixed. could be changed to flexible. 
        if self.num_dimensions > 3:
            self.gestalten = True
            if self.num_dimensions > 6:
                self.dir_mag_gest = True


        self.skelrenderer = SKEL_RENDERER()

        self.set_modification = False

        print('Initialized test procedure.')


    def set_dimensions(self, new_dimensions):
        self.num_dimensions = new_dimensions

        self.preprocessor.reset_dimensions(new_dimensions)
        # fixed. could be changed to flexible. 
        if self.num_dimensions > 3:
            self.gestalten = True
            if self.num_dimensions > 6:
                self.dir_mag_gest = True
            else:
                self.dir_mag_gest = False



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
        data_amc_path += ['Data_Compiler/S05T01.amc']
        data_asf_path += ['Data_Compiler/S05T01.asf']

        data_amc_path += ['Data_Compiler/S06T01.amc']
        data_asf_path += ['Data_Compiler/S06T01.asf']

        data_amc_path += ['Data_Compiler/S08T02.amc']
        data_asf_path += ['Data_Compiler/S08T02.asf']

        data_amc_path += ['Data_Compiler/S07T02.amc']
        data_asf_path += ['Data_Compiler/S07T02.asf']

        return data_amc_path, data_asf_path

    
    def load_data_all(self, asf_paths, amc_paths, sample_nums, modification):         
        data = []
        
        for i in range(len(sample_nums)): 
            if not self.set_modification:
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


        if len(set(sample_nums)) != 1:
            maxLen = max(sample_nums)
            for i in range(len(data)):
                (n, d, f) = data[i]
                if d.shape[0] < maxLen:
                    mult_factor = np.ceil(np.array([maxLen/d.shape[0]]))
                    mult_factor = mult_factor.astype(int)
                    for k in range(mult_factor[0]):
                        d = torch.cat([d,d])
                    d_new = d[:maxLen]
                    data[i] = (n, d_new, f)

        return data


    def load_data_original(self, asf_path, amc_path, num_samples): 
        if self.gestalten:
            observations, joint_names = self.preprocessor.get_AT_data_gestalten(asf_path, amc_path, num_samples)
        else:
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
                if self.gestalten:
                    if self.dir_mag_gest:
                        mag = data[:,:, -1].view(num_samples-1, self.num_observations, 1)
                        data = data[:,:, :-1]
                    data = torch.cat([
                        data[:,:, :self.num_dimensions], 
                        data[:,:, self.num_dimensions:]], dim=2)
                    data = data.view((num_samples-1)*self.num_observations*2, 3)
                    data = self.PERSP_TAKER.qrotate(data, modify)   
                    data = data.reshape(num_samples-1, self.num_observations, 6)
                    if self.dir_mag_gest:
                        data = torch.cat([data, mag], dim=2)
                else:
                    data = data.view(num_samples*self.num_observations, self.num_dimensions)
                    data = self.PERSP_TAKER.qrotate(data, modify)   
                    data = data.view(original_shape) 

                print("Q-Rotated", name)
                name += "_qrotated"

            elif mode == 'eulrotate':
                rotmat = self.PERSP_TAKER.compute_rotation_matrix_(modify[0], modify[1], modify[2])

                if self.gestalten:
                    if self.dir_mag_gest:
                        mag = data[:,:, -1].view(num_samples-1, self.num_observations, 1)
                        data = data[:,:, :-1]
                    data = torch.cat([
                        data[:,:, :self.num_dimensions], 
                        data[:,:, self.num_dimensions:]], dim=2)
                    data = data.view((num_samples-1)*self.num_observations*2, 3)
                    data = self.PERSP_TAKER.rotate(data, rotmat)   
                    data = data.reshape(num_samples-1, self.num_observations, 6)
                    if self.dir_mag_gest:
                        data = torch.cat([data, mag], dim=2)
                else:
                    data = data.view(num_samples*self.num_observations, self.num_dimensions)
                    data = self.PERSP_TAKER.rotate(data, rotmat)   
                    data = data.view(original_shape)

                
                print("Euler-Rotated", name)
                name += "_eulrotated"

            elif mode == 'translate':
                if self.gestalten:
                    non_pos = data[:,:,3:]
                    data = data[:,:,:3]
                    original_shape = data.size()
                    data = data.view((num_samples-1)*self.num_observations, 3)
                else:
                    original_shape = data.size()
                    data = data.view(num_samples*self.num_observations, self.num_dimensions)
                   
                data = self.PERSP_TAKER.translate(data, modify)
                data = data.view(original_shape)

                if self.gestalten:
                    data = torch.cat([data, non_pos], dim=2)

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

    
    def init_modification_params(self):
        self.new_order = None
        self.reorder = None
        self.new_rotation = None
        self.rerotate = None
        self.new_translation = None
        self.retranslate = None


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
        self.save_results_to_pt([final_predictions], ['final_predictions'])
        figures = []
        figures += [baptat.evaluator.plot_prediction_errors(results[0])]
        figures += [baptat.evaluator.plot_at_losses(results[1], 'History of overall losses during active tuning')]
        
        return results, figures


    def save_tesor_to_csv(self, tensor, path): 
        torch.save(tensor, path)


    def save_figures(self, figures, names):
        for i in range(len(figures)): 
            fig = figures[i]
            fig.savefig(self.result_path + names[i] + '.png')
            # fig.savefig(self.result_path + names[i] + '.png', bbox_inches='tight', dpi=150)

    
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

    
    def render(self, data):
        self.skelrenderer.render(data, None, None, False)
    
    
    def render_gestalt(self, data):
        pos = data[:,:,:3]
        dir = data[:,:,3:6]
        if self.dir_mag_gest:
            mag = data[:,:,-1]
        else:
            mag = torch.ones(data[:,:,1].shape)

        self.skelrenderer.render(pos, dir, mag, True)
  

    def run(): 
        pass


    def terminate(self):
        plt.close('all')
    
