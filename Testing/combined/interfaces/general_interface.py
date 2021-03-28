import numpy as np
import torch 
from torch import nn

import pandas as pd
import os

import sys

sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from Testing.TESTING_procedure_abstract import TEST_PROCEDURE
# from BAPTAT_5_combination_class_gestalten_decay import COMBI_BAPTAT_GESTALTEN
# from BAPTAT_5_combination_class_gestalten_clamp import COMBI_BAPTAT_GESTALTEN
from BAPTAT_5_combination_class_gestalten import COMBI_BAPTAT_GESTALTEN
from BinAndPerspTaking.perspective_taking import Perspective_Taker



class TESTER(TEST_PROCEDURE): 

    def __init__(self, num_features, num_observations, num_dimensions, experiment_name):
        super().__init__(num_features, num_observations, num_dimensions)
        experiment_path = "D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT/Grafics/GestaltRuns/"+experiment_name+'/'
        super().create_trial_directory(experiment_path)
        self.trial_path = self.result_path
        self.BAPTAT = COMBI_BAPTAT_GESTALTEN()
        self.PERSP_TAKER = Perspective_Taker(num_observations, num_dimensions, False, False)
        self.do_binding = False
        self.do_rotation = False
        self.rotation_type = None
        self.do_translation = False
        
        print('Initialized test environment.')


    def load_data(self, modifications=None, sample_nums=None):
        amc_paths, asf_paths = self.get_data_paths()
        self.init_modification_params()
        if self.gestalten: 
            dim_spare = self.num_dimensions
            self.num_dimensions = 3

        data_modification = []
        for action, modify, specify in modifications: 
            if action == 'bind':
                self.do_binding = True

                if modify == 'rand' or modify == 'det':
                    if modify == 'rand': 
                        modification = np.array(range(self.num_observations))
                        np.random.shuffle(modification)
                        print('Randomly modified order of observed features: ', modification)

                    elif modify == 'det': 
                        swap_pairs = [(2,4), (6, 10)]
                        modification = np.array(range(self.num_observations))
                        for i,j in swap_pairs:
                            modification[i], modification[j] = modification[j], modification[i]
                        print('Deterministically modified order of observed features: ', modification)

                    self.new_order = modification
                    self.reorder = torch.tensor([np.where(self.new_order == i)[0][0] 
                                                            for i in range(self.num_observations)])
                    modification = torch.tensor(modification, dtype=torch.int64)
                    data_modification += [('rebind', modification)]
                else: 
                    modification = None
                    self.new_order = None
                    self.reorder = None

            elif action == 'rotate': 
                self.do_rotation = True
                self.rotation_type = specify

                if modify == 'rand' or modify == 'det':
                    if modify == 'rand': 
                        if specify == 'qrotate':
                            modification = torch.rand(1,4) 
                            modification = self.PERSP_TAKER.norm_quaternion(modification)
                        elif specify == 'eulrotate': 
                            modification = torch.rand(3).view(3,1)*360

                        print(f'Randomly modified rotation of observed features: {specify} by {modification}')

                    elif modify == 'det': 
                        modification = torch.Tensor([0.5,0.4,0.3,0.2]).view(1,4)
                        modification = self.PERSP_TAKER.norm_quaternion(modification)
                        if specify == 'eulrotate': 
                            modification = torch.rad2deg(self.PERSP_TAKER.qeuler(modification,'zyx').view(3,1))
                        
                        print(f'Deterministically modified rotation of observed features: {specify} by {modification}')

                    self.new_rotation = modification
                    if specify == 'qrotate':
                        self.rerotate = self.PERSP_TAKER.inverse_rotation_quaternion(modification)
                        self.rerotation_matrix = self.PERSP_TAKER.quaternion2rotmat(self.rerotate)

                    elif specify == 'eulrotate': 
                        self.rerotate = self.PERSP_TAKER.inverse_rotation_angles(modification)
                        self.rerotation_matrix = self.PERSP_TAKER.compute_rotation_matrix_(
                            self.rerotate[0], self.rerotate[1], self.rerotate[2]).view(3,3)

                    data_modification += [(specify, modification)]
                else: 
                    modification = None
                    self.new_rotation = None
                    if specify == 'qrotate':
                        self.rerotate = torch.zeros(1,4)
                        self.rerotate[0,0] = 1.0
                    elif specify == 'eulrotate':
                        self.rerotate = torch.zeros(3).view(3,1)
                    self.rerotation_matrix = torch.Tensor(np.identity(self.num_dimensions))

            elif action == 'translate': 
                self.do_translation = True

                if modify == 'rand' or modify == 'det':
                    if modify == 'rand': 
                        lower_bound = specify[0] * torch.ones(self.num_dimesions)
                        upper_bound = specify[1] * torch.ones(self.num_dimesions)
                        range_size = upper_bound - lower_bound
                        modification = torch.mul(torch.rand(self.num_dimensions), range_size)
                        print(f'Randomly modified translation of observed features: {modification}')

                    elif modify == 'det': 
                        modification = torch.Tensor([3.2, -2.6, 0.4])
                        print(f'Deterministically modified translation of observed features: {modification}')

                    self.new_translation = modification
                    self.retranslate = self.PERSP_TAKER.inverse_translation_bias(modification)

                    data_modification += [('translate', modification)]
                else: 
                    modification = None
                    self.new_translation = None
                    self.retranslate = torch.zeros(self.num_dimensions)
                    print(self.num_dimensions)
            
            else: 
                print('Received unknown modification. Skipped.')

        if self.gestalten: 
            self.num_dimensions = dim_spare
        
        if data_modification == []:
            data_modification = None
        data = self.load_data_all(asf_paths, amc_paths, sample_nums, data_modification)

        # dt = torch.cat([f for (_, f, _) in data])
        # self.render_gestalt(dt)
        # exit()
        
        return data


    def prepare_inference(self, 
        rotation_type, 
        current_rotation,
        current_translation, 
        num_frames, 
        model_path, 
        tuning_length, 
        num_tuning_cycles, 
        at_loss_function, 
        at_loss_parameters, 
        at_learning_rates,
        at_learning_rate_state, 
        at_momenta):


        super().prepare_inference(
            self.BAPTAT, 
            num_frames, 
            model_path, 
            tuning_length, 
            num_tuning_cycles, 
            at_loss_function, 
            at_learning_rates, 
            at_learning_rate_state, 
            at_momenta)

        # set ideal comparision parameters
        ideal_binding, ideal_rotation, ideal_translation = None, None, None
        info_string = ''

        # binding
        if self.do_binding:
            # NOTE: ideal matrix is always identity, bc then the FBE and determinant can be calculated => provide reorder
            ideal_binding = torch.Tensor(np.identity(self.num_features))

            info_string += f' - modification of binding order: \t{self.new_order}\n\n'
            info_string += f' - modification of body rotation with {rotation_type} by \n\t{self.new_rotation if current_rotation is not None else None}\n'
        
        # rotation
        if self.do_rotation:
            if current_rotation is not None: 
                rerot = self.rerotate
                rerot_matrix = self.rerotation_matrix
            else: 
                if rotation_type == 'qrotate':
                    rerot = torch.zeros(1,4)
                    rerot[0,0] = 1.0
                elif rotation_type == 'eulrotate':
                    rerot = torch.zeros(3).view(3,1)
                rerot_matrix = torch.Tensor(np.identity(3))
                # rerot_matrix = torch.Tensor(np.identity(self.num_dimensions))

            ideal_rotation = (rerot, rerot_matrix)

            info_string += f' - optimally infered rotation: \n\t{rerot}\n\n'

        # translation
        if self.do_translation:
            if current_translation is not None: 
                retrans = self.retranslate
            else: 
                retrans = torch.zeros(3)
                # retrans = torch.zeros(self.num_dimensions)

            ideal_translation = retrans       

            info_string += f' - modification of body translation: {self.new_translation if current_translation is not None else None}\n'
            info_string += f' - optimally infered translation: \n\t{retrans}\n\n'
        
        self.BAPTAT.set_comparison_values(ideal_binding, ideal_rotation, ideal_translation)
        
        info_string = self.construct_info_string(info_string, at_loss_parameters)
        info_string += f' - learning rates:\n' 
        info_string += f'\t> binding: \t\t{self.BAPTAT.at_learning_rate_binding}\n'
        info_string += f'\t> rotation: \t\t{self.BAPTAT.at_learning_rate_rotation}\n'
        info_string += f'\t> translation: \t\t{self.BAPTAT.at_learning_rate_translation}\n'
        info_string += f' - momenta:\n' 
        info_string += f'\t> binding: \t\t{self.BAPTAT.bm_momentum}\n'
        info_string += f'\t> rotation: \t\t{self.BAPTAT.r_momentum}\n'
        info_string += f'\t> translation: \t\t{self.BAPTAT.c_momentum}\n'
        
        self.write_to_file(info_string, self.result_path+'parameter_information.txt')
        print('Ready to run AT inference for binding task! \nInitialized parameters with: \n' + info_string)
        

    def evaluate(self, 
        observations, 
        final_predictions, 
        final_binding_matrix,
        final_binding_entries, 
        final_rotation_values, 
        final_rotation_matrix, 
        final_translation_values, 
        feature_names, 
        order):

        results, figures = super().evaluate(self.BAPTAT, observations, final_predictions)
        res_i = 2

        fig_names = ['prediction_errors', 'at_loss_history']
        res_names = []
        csv_names = []
        csv_names += fig_names
        pt_results = []

        ## Save figures
        if self.do_binding: 
            fig_names += ['determinante_history']
            csv_names += ['determinante_history']
            figures += [self.BAPTAT.evaluator.plot_at_losses(results[res_i], 'History of binding matrix determinante')]
            res_i += 1
        
            if self.num_features != self.num_observations: 
                figures += [self.BAPTAT.evaluator.plot_at_losses(
                    results[res_i][:,0], 
                    'History of binding matrix loss (FBE) for cleared matrix'
                )]
                figures += [self.BAPTAT.evaluator.plot_at_losses(
                    results[res_i][:,1], 
                    'History of binding matrix loss (FBE) for outcast line and additional cloumns'
                )]
                figures += [self.BAPTAT.evaluator.plot_at_losses(
                    results[res_i][:,2], 
                    'History of binding matrix loss (FBE) for whole matrix'
                )]
                res_i += 1

                figures += [self.BAPTAT.evaluator.plot_binding_matrix_nxm(
                    final_binding_matrix, 
                    feature_names, 
                    self.num_observations,
                    self.BAPTAT.get_additional_features(),
                    'Binding matrix showing relative contribution of observed feature to input feature', 
                    order
                )]
                figures += [self.BAPTAT.evaluator.plot_binding_matrix_nxm(
                    final_binding_entries, 
                    feature_names, 
                    self.num_observations,
                    self.BAPTAT.get_additional_features(),
                    'Binding matrix entries showing contribution of observed feature to input feature', 
                    order
                )]
                figures += [self.BAPTAT.evaluator.plot_outcast_gradients(
                    self.BAPTAT.get_oc_grads(), 
                    feature_names, 
                    self.num_observations,
                    self.BAPTAT.get_additional_features(),
                    'Gradients of outcast line for observed features during inference'
                )]
                fig_names += [
                    'fbe_cleared_history', 'fbe_oc+af_history', 'fbe_whole_history', 
                    'final_binding_matirx', 'final_binding_neurons_activities','outcat_line_gradients']
                csv_names += ['fbe_cleared_history', 'fbe_oc+af_history', 'fbe_whole_history', 'outcat_line_gradients']

            else:
                figures += [self.BAPTAT.evaluator.plot_at_losses(results[res_i], 'History of binding matrix loss (FBE)')]
                res_i += 1

                figures += [self.BAPTAT.evaluator.plot_binding_matrix(
                    final_binding_matrix, 
                    feature_names, 
                    'Binding matrix showing relative contribution of observed feature to input feature', 
                    order
                )]
                figures += [self.BAPTAT.evaluator.plot_binding_matrix(
                    final_binding_entries, 
                    feature_names, 
                    'Binding matrix entries showing contribution of observed feature to input feature', 
                    order
                )]
                fig_names += ['fbe_history', 'final_binding_matirx', 'final_binding_neurons_activities']
                csv_names += ['fbe_history']
            
            res_names += ['final_binding_matirx', 'final_binding_neurons_activities']
            pt_results += [final_binding_matrix, final_binding_entries]
        else:
            results = results[:2] + results[4:]

        if self.do_rotation:
            figures += [self.BAPTAT.evaluator.plot_at_losses(results[res_i], 'History of rotation matrix loss (MSE)')]
            res_i += 1
            figures += [self.BAPTAT.evaluator.plot_at_losses(results[res_i], 'History of rotation values loss')]
            res_i += 1

            fig_names += ['rotmat_loss_history', 'rotval_loss_history']
            csv_names += ['rotmat_loss_history', 'rotval_loss_history']
            res_names += ['final_rotation_values', 'final_rotation_matrix']
            pt_results += [final_rotation_values, final_rotation_matrix]
        else:
            results = results[:-3] + results[-1:]
        
        if self.do_translation:
            figures += [self.BAPTAT.evaluator.plot_at_losses(results[res_i], 'History of translation loss (MSE)')]
            res_i += 1

            fig_names += ['transba_loss_history']
            csv_names += ['transba_loss_history']
            res_names += ['final_translation_values']
            pt_results += [final_translation_values]
        else: 
            results = results[:-1]


        self.save_figures(figures, fig_names)
        self.save_results_to_csv(results, csv_names)
        self.save_results_to_pt(pt_results, res_names)

        return results, csv_names


    def run(self, 
        experiment_dir,
        modification,
        sample_nums, 
        model_path, 
        tuning_length, 
        num_tuning_cycles, 
        at_loss_function,
        loss_parameters,
        at_learning_rates, 
        at_learning_rate_state, 
        at_momenta, 
        grad_calculations):

        print('*************************************************************************************')

        experiment_results = []
        self.BAPTAT.set_dimensions(self.num_dimensions)
        print(f'Use model: {model_path}')

        data = self.load_data(modification, sample_nums)

        res_path = ""
        if self.do_binding:
            res_path += 'b_'
        if self.do_rotation:
            res_path += 'r_'
        if self.do_translation:
            res_path += 't_'
        self.prefix_res_path = self.trial_path + res_path
        res_path = self.prefix_res_path + experiment_dir
        if experiment_dir != "":
            os.mkdir(res_path)
            print('Created directory: '+ res_path)

        sample_names = []
        for (name, observations, feat_names) in data:
            # name = d[0]
            # observations = d[1]
            # feat_names = d[2]
            sample_names += [name]
            obs_shape = observations.shape
            num_frames = observations.size()[0]
            self.BAPTAT.set_data_parameters_(
                num_frames, self.num_observations, self.num_features, self.num_dimensions)


            self.result_path = res_path+name+'/'
            os.mkdir(self.result_path)

            new_order = self.new_order
            new_rotation = self.new_rotation
            new_translation = self.new_translation
            if new_order is not None and '_' not in name:
                new_order = None
                new_rotation = None
                new_translation = None
            
            self.prepare_inference(
                self.rotation_type, 
                new_rotation,
                new_translation, 
                num_frames, 
                model_path, 
                tuning_length, 
                num_tuning_cycles, 
                at_loss_function, 
                loss_parameters, 
                at_learning_rates,
                at_learning_rate_state, 
                at_momenta)

            # self.render_gestalt(observations)
            # self.render_gestalt(data[1][1])
            # exit()

            [at_final_inputs,
                at_final_predictions, 
                final_binding_matrix,
                final_binding_entries, 
                final_rotation_values, 
                final_rotation_matrix, 
                final_translation_values 
                ] = self.BAPTAT.run_inference(
                    observations, 
                    grad_calculations, 
                    self.do_binding, 
                    self.do_rotation,
                    self.do_translation,
                    new_order, 
                    self.reorder)

            # if self.gestalten:
            #     self.render_gestalt(at_final_inputs.view(num_frames, self.num_features, self.num_dimensions))
            #     self.render_gestalt(at_final_predictions.view(num_frames, self.num_features, self.num_dimensions))
            # else:
            #     self.render(at_final_inputs.view(num_frames, self.num_features, self.num_dimensions))
            #     self.render(at_final_predictions.view(num_frames, self.num_features, self.num_dimensions))
            self.save_results_to_pt([at_final_inputs], ['final_inputs'])

            # reorder observations to compare with final predictions
            if new_order is not None:
                observations = self.order(observations, self.reorder)

            # rerotate observations to compare with final predictions 
            if new_rotation is not None:
                if self.dir_mag_gest:
                    mag = observations[:,:, -1].view(num_frames, self.num_observations, 1)
                    observations = observations[:,:, :-1]
                observations = torch.cat([
                    observations[:,:, :self.num_dimensions], 
                    observations[:,:, self.num_dimensions:]], dim=2)
                observations = observations.view((num_frames)*self.num_observations*2, 3)

                if self.rotation_type == 'qrotate':
                    observations = self.PERSP_TAKER.qrotate(observations, self.rerotate)   
                else:
                    rotmat = self.PERSP_TAKER.compute_rotation_matrix_(self.rerotate[0], self.rerotate[1], self.rerotate[2])
                    observations = self.PERSP_TAKER.rotate(observations, rotmat)   

                observations = observations.reshape(num_frames, self.num_observations, 6)
                if self.dir_mag_gest:
                    observations = torch.cat([observations, mag], dim=2)
                

            # retranslate observations to compare with final predictions 
            if new_translation is not None:
                non_pos = observations[:,:,3:]
                observations = observations[:,:,:3]
                observations = observations.view((num_frames)*self.num_observations, 3)
                self.PERSP_TAKER.translate(observations, self.retranslate)
                observations = observations.view((num_frames), self.num_observations, 3)
                observations = torch.cat([observations, non_pos], dim=2)


            res, res_names = self.evaluate(
                observations, 
                at_final_predictions, 
                final_binding_matrix,
                final_binding_entries, 
                final_rotation_values, 
                final_rotation_matrix, 
                final_translation_values, 
                feat_names, 
                new_order)
            print('Evaluated current run.')


            super().terminate()
            print('Terminated current run.')


            experiment_results += [[name, res, final_binding_matrix, final_binding_entries]]

        return sample_names, res_names, experiment_results
      
        