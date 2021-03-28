# packet imports 
import numpy as np
from numpy.lib.function_base import append 
import torch
import copy
from torch import nn, autograd
from torch.autograd import Variable
from torch._C import device
import matplotlib.pyplot as plt

# class imports 
from BinAndPerspTaking.binding_nxm import BINDER_NxM
from CoreLSTM.core_lstm import CORE_NET
from Data_Compiler.data_preparation import Preprocessor
from BAPTAT_evaluation import BAPTAT_evaluator


class SEP_BINDING_GESTALTEN():

    def __init__(self):
        ## General parameters 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autograd.set_detect_anomaly(True)

        torch.set_printoptions(precision=8)

        
        ## Set default parameters
        ##   -> Can be changed during experiments 

        self.scale_mode = 'rcwSM'
        self.scale_combo = 'comp_mult'
        self.grad_bias = 1.5
        self.nxm = False
        self.additional_features = None
        self.nxm_enhance = 'square', 
        self.nxm_last_line_scale = 0.1
        self.dummie_init = 0.1



    ############################################################################
    ##########  PARAMETERS  ####################################################

    def set_scale_mode(self, mode): 
        self.scale_mode = mode
        print('Reset scale mode: ' + self.scale_mode)


    def set_scale_combination(self, combination): 
        self.scale_combo = combination
        print('Reset scale combination: ' + self.scale_combo)


    def set_additional_features(self, index_addition): 
        self.additional_features = index_addition
        print(f'Additional features to the LSTM-input at indices {self.additional_features}')


    def set_nxm_enhancement(self, enhancement): 
        self.nxm_enhance = enhancement
        print(f'Enhancement for outcast line: {self.nxm_enhance}')


    def set_nxm_last_line_scale(self, scale_factor): 
        self.nxm_last_line_scale = scale_factor
        print(f'Scaler for outcast line: {self.nxm_last_line_scale}')


    def set_dummie_init_value(self, init_value):
        self.dummie_init = init_value
        print(f'Initial value for dummie line: {self.dummie_init}')


    def set_weighted_gradient_bias(self, bias):
        # bias > 1 => favor recent
        # bias < 1 => favor earlier
        self.grad_bias = bias
        print(f'Reset bias for gradient weighting: {self.grad_bias}')

    
    def set_data_parameters_(self, 
        num_frames, 
        num_observations, 
        num_input_features, 
        num_input_dimesions): 

        ## Define data parameters
        self.num_frames = num_frames
        self.num_observations = num_observations
        self.num_input_features = num_input_features
        self.num_input_dimensions = num_input_dimesions
        self.input_per_frame = self.num_input_features * self.num_input_dimensions
        self.nxm = (self.num_observations != self.num_input_features)
        
        # fixed gestalten definition. could be changed to flexible.
        if self.num_input_dimensions == 7:
            self.gestalten = True
            self.num_spatial_dimensions = self.num_input_dimensions-4
            self.num_pos = self.num_spatial_dimensions
            self.num_dir = self.num_spatial_dimensions
            self.num_mag = 1
        else:
            self.num_spatial_dimensions = self.num_input_dimensions
        
        self.binder = BINDER_NxM(
            num_observations=self.num_observations, 
            num_features=self.num_input_features, 
            gradient_init=True)

        self.preprocessor = Preprocessor(
            self.num_observations, 
            self.num_input_features, 
            self.num_input_dimensions)

        self.evaluator = BAPTAT_evaluator(
            self.num_frames, 
            self.num_observations, 
            self.num_input_features, 
            self.preprocessor)

    
    def set_tuning_parameters_(self, 
        tuning_length, 
        num_tuning_cycles, 
        loss_function, 
        at_learning_rate_binding, 
        at_learning_rate_state, 
        at_momentum_binding):
         
        ## Define tuning parameters 
        self.tuning_length = tuning_length          # length of tuning horizon 
        self.tuning_cycles = num_tuning_cycles      # number of tuning cycles in each iteration 

        # possible loss functions
        self.at_loss = loss_function
        self.mse = nn.MSELoss()
        self.l1Loss = nn.L1Loss()
        self.smL1Loss = nn.SmoothL1Loss(reduction='sum')
        self.l2Loss = lambda x,y: self.mse(x, y) * (self.num_input_dimensions * self.num_input_features)

        # define learning parameters 
        self.at_learning_rate = at_learning_rate_binding
        self.at_learning_rate_state = at_learning_rate_state
        self.bm_momentum = at_momentum_binding
        self.at_loss_function = self.mse

        print('Parameters set.')


    def get_additional_features(self): 
        return self.additional_features 

    
    def get_oc_grads(self): 
        return self.oc_grads 


    def init_model_(self, model_path): 
        ## Load model
        self.core_model = CORE_NET(input_size=105, hidden_layer_size=210)
        self.core_model.load_state_dict(torch.load(model_path))
        self.core_model.eval()
        self.core_model.to(self.device)

        print('Model loaded.')


    def init_inference_tools(self):
        ## Define tuning variables
        # general
        self.obs_count = 0
        self.at_inputs = torch.tensor([]).to(self.device)
        self.at_predictions = torch.tensor([]).to(self.device)
        self.at_final_predictions = torch.tensor([]).to(self.device)
        self.at_losses = []

        # state 
        self.at_states = []
        # state_optimizer = torch.optim.Adam(init_state, at_learning_rate)

        # binding
        self.ideal_binding = torch.Tensor(np.identity(self.num_input_features)).to(self.device)

        self.Bs = []
        self.B_grads = [None] * (self.tuning_length+1)
        self.B_upd = [None] * (self.tuning_length+1)
        self.bm_losses = []
        self.bm_dets = []

        self.oc_grads = []

    
    def set_comparison_values(self, ideal_binding):
        self.ideal_binding = ideal_binding.to(self.device)
        if self.nxm:
            self.ideal_binding = self.binder.ideal_nxm_binding(
                self.additional_features, self.ideal_binding).to(self.device)



    ############################################################################
    ##########  INFERENCE  #####################################################
    
    def run_inference(self, observations, grad_calculation, order, reorder):

        if reorder is not None:
            reorder = reorder.to(self.device)
        
        at_final_predictions = torch.tensor([]).to(self.device)
        at_final_inputs = torch.tensor([]).to(self.device)

        ## Binding matrices 
        # Init binding entries 
        bm = self.binder.init_binding_matrix_det_()
        # bm = binder.init_binding_matrix_rand_()
        # print(bm)
        dummie_line = torch.ones(1,self.num_observations).to(self.device) * self.dummie_init

        for i in range(self.tuning_length+1):
            matrix = bm.clone().to(self.device)
            if self.nxm:
                matrix = torch.cat([matrix, dummie_line])
            matrix.requires_grad_()
            self.Bs.append(matrix)
            
        # print(f'BMs different in list: {self.Bs[0] is not self.Bs[1]}')

        ## Core state
        # define scaler
        state_scaler = 0.95

        # init state
        at_h = torch.zeros(1, self.core_model.hidden_size).to(self.device)
        at_c = torch.zeros(1, self.core_model.hidden_size).to(self.device)

        at_h.requires_grad = True
        at_c.requires_grad = True

        init_state = (at_h, at_c)
        state = (init_state[0], init_state[1])

        ############################################################################
        ##########  FORWARD PASS  ##################################################

        for i in range(self.tuning_length):
            o = observations[self.obs_count].to(self.device)
            self.at_inputs = torch.cat((self.at_inputs, o.reshape(1, self.num_observations, self.num_input_dimensions)), 0)
            self.obs_count += 1

            bm = self.binder.scale_binding_matrix(
                self.Bs[i], 
                self.scale_mode, 
                self.scale_combo, 
                self.nxm_enhance, 
                self.nxm_last_line_scale)
            if self.nxm:
                bm = bm[:-1]
            x_B = self.binder.bind(o, bm)
            
            x = self.preprocessor.convert_data_AT_to_LSTM(x_B)

            state = (state[0] * state_scaler, state[1] * state_scaler)
            new_prediction, state = self.core_model(x, state)  
            self.at_states.append(state)
            self.at_predictions = torch.cat((self.at_predictions, new_prediction.reshape(1,self.input_per_frame)), 0)

        ############################################################################
        ##########  ACTIVE TUNING ##################################################

        while self.obs_count < self.num_frames:
            # TODO folgendes evtl in function auslagern
            o = observations[self.obs_count].to(self.device)
            self.obs_count += 1

            bm = self.binder.scale_binding_matrix(
                self.Bs[-1], 
                self.scale_mode, 
                self.scale_combo,
                self.nxm_enhance, 
                self.nxm_last_line_scale)
            if self.nxm:
                bm = bm[:-1]
            x_B = self.binder.bind(o, bm)
            
            x = self.preprocessor.convert_data_AT_to_LSTM(x_B)

            ## Generate current prediction 
            with torch.no_grad():
                state = self.at_states[-1]
                state = (state[0] * state_scaler, state[1] * state_scaler)
                new_prediction, state = self.core_model(x, state)

            ## For #tuning_cycles 
            for cycle in range(self.tuning_cycles):
                print('----------------------------------------------')

                # Get prediction
                p = self.at_predictions[-1]

                # Calculate error 
                loss = self.at_loss(p,x[0])

                # Propagate error back through tuning horizon 
                loss.backward(retain_graph = True)

                self.at_losses.append(loss.clone().detach().cpu().numpy())
                print(f'frame: {self.obs_count} cycle: {cycle} loss: {loss}')

                # Update parameters 
                with torch.no_grad():
                    
                    # Calculate gradients with respect to the entires 
                    for i in range(self.tuning_length+1):
                        self.B_grads[i] = self.Bs[i].grad

                    # print(B_grads[tuning_length])
                    
                    
                    # Calculate overall gradients 
                    if grad_calculation == 'lastOfTunHor':
                        ### version 1
                        grad_B = self.B_grads[0]
                    elif grad_calculation == 'meanOfTunHor':
                        ### version 2 / 3
                        grad_B = torch.mean(torch.stack(self.B_grads), dim=0)
                    elif grad_calculation == 'weightedInTunHor':
                        ### version 4
                        weighted_grads_B = [None] * (self.tuning_length+1)
                        for i in range(self.tuning_length+1):
                            weighted_grads_B[i] = np.power(self.grad_bias, i) * self.B_grads[i]
                        grad_B = torch.mean(torch.stack(weighted_grads_B), dim=0)
                    
                    # print(f'grad_B: {grad_B}')
                    # print(f'grad_B: {torch.norm(grad_B, 1)}')
                    

                    # Update parameters in time step t-H with saved gradients 
                    grad_B = grad_B.to(self.device)
                    upd_B = self.binder.decay_update_binding_matrix_(self.Bs[0], grad_B, self.at_learning_rate, self.bm_momentum)
                    # upd_B = self.binder.update_binding_matrix_(self.Bs[0], grad_B, self.at_learning_rate, self.bm_momentum)

                    # Compare binding matrix to ideal matrix
                    # NOTE: ideal matrix is always identity, bc then the FBE and determinant can be calculated => provide reorder
                    c_bm = self.binder.scale_binding_matrix(upd_B, self.scale_mode, self.scale_combo)
                    if order is not None: 
                        c_bm = c_bm.gather(1, reorder.unsqueeze(0).expand(c_bm.shape))

                    if self.nxm:
                        self.oc_grads.append(grad_B[-1])
                        FBE = self.evaluator.FBE_nxm_additional_features(c_bm, self.ideal_binding, self.additional_features)
                        c_bm = self.evaluator.clear_nxm_binding_matrix(c_bm, self.additional_features)
                    
                    mat_loss = self.evaluator.FBE(c_bm, self.ideal_binding)

                    if self.nxm:
                        mat_loss = torch.stack([mat_loss, FBE, mat_loss+FBE])
                    self.bm_losses.append(mat_loss)
                    print(f'loss of binding matrix (FBE): {mat_loss}')

                    # Compute determinante of binding matrix
                    det = torch.det(c_bm)
                    self.bm_dets.append(det)
                    print(f'determinante of binding matrix: {det}')


                    # Zero out gradients for all parameters in all time steps of tuning horizon
                    for i in range(self.tuning_length+1):
                        self.Bs[i].requires_grad = False
                        self.Bs[i].grad.data.zero_()

                    # Update all parameters for all time steps 
                    for i in range(self.tuning_length+1):
                        self.Bs[i].data = upd_B.clone().data
                        self.Bs[i].requires_grad = True
                    
                    # print(Bs[0])


                    # Initial state
                    g_h = at_h.grad.to(self.device)
                    g_c = at_c.grad.to(self.device)

                    upd_h = init_state[0] - self.at_learning_rate_state * g_h
                    upd_c = init_state[1] - self.at_learning_rate_state * g_c

                    at_h.data = upd_h.clone().detach().requires_grad_()
                    at_c.data = upd_c.clone().detach().requires_grad_()

                    at_h.grad.data.zero_()
                    at_c.grad.data.zero_()

                    # print(f'updated init_state: {init_state}')
                
                ## REORGANIZE FOR MULTIPLE CYCLES!!!!!!!!!!!!!

                # forward pass from t-H to t with new parameters 
                init_state = (at_h, at_c)
                state = (init_state[0], init_state[1])
                self.at_predictions = torch.tensor([]).to(self.device)
                for i in range(self.tuning_length):

                    bm = self.binder.scale_binding_matrix(
                        self.Bs[i], 
                        self.scale_mode, 
                        self.scale_combo, 
                        self.nxm_enhance, 
                        self.nxm_last_line_scale)
                    if self.nxm:
                        bm = bm[:-1]
                    x_B = self.binder.bind(self.at_inputs[i], bm)
                    x = self.preprocessor.convert_data_AT_to_LSTM(x_B)

                    state = (state[0] * state_scaler, state[1] * state_scaler)
                    upd_prediction, state = self.core_model(x, state)
                    self.at_predictions = torch.cat((self.at_predictions, upd_prediction.reshape(1,self.input_per_frame)), 0)
                    
                    # for last tuning cycle update initial state to track gradients 
                    if cycle==(self.tuning_cycles-1) and i==0: 
                        with torch.no_grad():
                            final_prediction = self.at_predictions[0].clone().detach().to(self.device)
                            final_input = x.clone().detach().to(self.device)

                        at_h = state[0].clone().detach().requires_grad_().to(self.device)
                        at_c = state[1].clone().detach().requires_grad_().to(self.device)
                        init_state = (at_h, at_c)
                        state = (init_state[0], init_state[1])

                    self.at_states[i] = state 

                # Update current binding
                bm = self.binder.scale_binding_matrix(
                    self.Bs[-1], 
                    self.scale_mode, 
                    self.scale_combo, 
                    self.nxm_enhance, 
                    self.nxm_last_line_scale)
                if self.nxm:
                    bm = bm[:-1]
                x_B = self.binder.bind(o, bm)
                x = self.preprocessor.convert_data_AT_to_LSTM(x_B)


            # END tuning cycle        

            ## Generate updated prediction 
            state = self.at_states[-1]
            state = (state[0] * state_scaler, state[1] * state_scaler)
            new_prediction, state = self.core_model(x, state)

            ## Reorganize storage variables            
            # observations
            at_final_inputs = torch.cat(
                (at_final_inputs, 
                final_input.reshape(1,self.input_per_frame)), 0)
            self.at_inputs = torch.cat(
                (self.at_inputs[1:], 
                o.reshape(1, self.num_observations, self.num_input_dimensions)), 0)
            
            # predictions
            at_final_predictions = torch.cat(
                (at_final_predictions, 
                final_prediction.reshape(1,self.input_per_frame)), 0)
            self.at_predictions = torch.cat(
                (self.at_predictions[1:], 
                new_prediction.reshape(1,self.input_per_frame)), 0)

        # END active tuning
        
        # store rest of predictions in at_final_predictions
        for i in range(self.tuning_length): 
            at_final_predictions = torch.cat(
                (at_final_predictions, 
                self.at_predictions[i].reshape(1,self.input_per_frame)), 0)
            x_i = self.binder.bind(self.at_inputs[i], bm)
            at_final_inputs = torch.cat(
                (at_final_inputs, 
                x_i.reshape(1,self.input_per_frame)), 0)

        # get final binding matrix
        final_binding_matrix = self.binder.scale_binding_matrix(self.Bs[-1].clone().detach(), self.scale_mode, self.scale_combo)
        # print(f'final binding matrix: {final_binding_matrix}')
        final_binding_entries = self.Bs[-1].clone().detach()
        # print(f'final binding entires: {final_binding_entries}')

        return at_final_inputs, at_final_predictions, final_binding_matrix, final_binding_entries




    ############################################################################
    ##########  EVALUATION #####################################################

    def get_result_history(
        self, 
        observations, 
        at_final_predictions):

        if self.nxm:
            pred_errors = self.evaluator.prediction_errors_nxm(
                observations, 
                self.additional_features, 
                self.num_observations, 
                at_final_predictions, 
                self.mse
            )
            self.bm_losses = torch.stack(self.bm_losses)

        else:
            pred_errors = self.evaluator.prediction_errors(
                observations, 
                at_final_predictions, 
                self.mse)

        return [pred_errors, self.at_losses, self.bm_dets, self.bm_losses]

