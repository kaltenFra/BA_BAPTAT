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
from BinAndPerspTaking.binding import Binder
from BinAndPerspTaking.binding_exmat import BinderExMat
from BinAndPerspTaking.perspective_taking import Perspective_Taker
from CoreLSTM.core_lstm import CORE_NET
from Data_Compiler.data_preparation import Preprocessor
from BAPTAT_evaluation import BAPTAT_evaluator


class SEP_BINDING():

    def __init__(self):
        ## General parameters 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autograd.set_detect_anomaly(True)

        torch.set_printoptions(precision=8)

        
        ## Set default parameters
        ##   -> Can be changed during experiments 

        self.scale_mode = 'rcwSM'
        self.scale_combo = 'comp_mult'



    ############################################################################
    ##########  PARAMETERS  ####################################################

    def set_scale_mode(self, mode): 
        self.scale_mode = mode
        print('Reset scale mode: ' + self.scale_mode)


    def set_scale_combination(self, combination): 
        self.scale_combo = combination
        print('Reset scale combination: ' + self.scale_combo)

    
    def set_data_parameters_(self, num_frames, num_input_features, num_input_dimesions): 
        ## Define data parameters
        self.num_frames = num_frames
        self.num_input_features = num_input_features
        self.num_input_dimensions = num_input_dimesions
        self.input_per_frame = self.num_input_features * self.num_input_dimensions
        
        self.preprocessor = Preprocessor(self.num_input_features, self.num_input_dimensions)
        self.evaluator = BAPTAT_evaluator(self.num_frames, self.num_input_features, self.preprocessor)
        
    
    def set_tuning_parameters_(self, tuning_length, num_tuning_cycles, loss_function, at_learning_rate_binding, at_learning_rate_state, at_momentum_binding): 
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



    def init_model_(self, model_path): 
        ## Load model
        self.core_model = CORE_NET()
        self.core_model.load_state_dict(torch.load(model_path))
        self.core_model.eval()
        # print('\nModel loaded:')
        # print(self.core_model)


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
        self.binder = BinderExMat(num_features=self.num_input_features, gradient_init=True)
        self.ideal_binding = torch.Tensor(np.identity(self.num_input_features)).to(self.device)

        self.Bs = []
        self.B_grads = [None] * (self.tuning_length+1)
        self.B_upd = [None] * (self.tuning_length+1)
        self.bm_losses = []
        self.bm_dets = []

    
    def set_comparison_values(self, ideal_binding):
        self.ideal_binding = ideal_binding


    ############################################################################
    ##########  INFERENCE  #####################################################
    
    def run_inference(self, observations, order, reorder):
        
        at_final_predictions = torch.tensor([]).to(self.device)

        ## Binding matrices 
        # Init binding entries 
        bm = self.binder.init_binding_matrix_det_()
        # bm = binder.init_binding_matrix_rand_()
        # print(bm)

        for i in range(self.tuning_length+1):
            matrix = bm.clone()
            matrix.requires_grad_()
            self.Bs.append(matrix)
            
        # print(f'BMs different in list: {self.Bs[0] is not self.Bs[1]}')

        ## Core state
        # define scaler
        state_scaler = 0.95

        # init state
        at_h = torch.zeros(1, self.core_model.hidden_size).requires_grad_().to(self.device)
        at_c = torch.zeros(1, self.core_model.hidden_size).requires_grad_().to(self.device)

        init_state = (at_h, at_c)
        self.at_states.append(init_state)
        state = (init_state[0], init_state[1])

        ############################################################################
        ##########  FORWARD PASS  ##################################################

        for i in range(self.tuning_length):
            o = observations[self.obs_count]
            self.at_inputs = torch.cat((self.at_inputs, o.reshape(1, self.num_input_features, self.num_input_dimensions)), 0)
            self.obs_count += 1

            bm = self.binder.scale_binding_matrix(self.Bs[i], self.scale_mode, self.scale_combo)
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
            o = observations[self.obs_count]
            self.obs_count += 1

            bm = self.binder.scale_binding_matrix(self.Bs[-1], self.scale_mode, self.scale_combo)
            x_B = self.binder.bind(o, bm)
            
            x = self.preprocessor.convert_data_AT_to_LSTM(x_B)

            ## Generate current prediction 
            with torch.no_grad():
                state = (state[0] * state_scaler, state[1] * state_scaler)
                new_prediction, state_new = self.core_model(x, self.at_states[-1])

            ## For #tuning_cycles 
            for cycle in range(self.tuning_cycles):
                print('----------------------------------------------')

                # Get prediction
                p = self.at_predictions[-1]

                # Calculate error 
                loss = self.at_loss(p,x[0])

                self.at_losses.append(loss.clone().detach().numpy())
                print(f'frame: {self.obs_count} cycle: {cycle} loss: {loss}')

                # Propagate error back through tuning horizon 
                loss.backward(retain_graph = True)

                # Update parameters 
                with torch.no_grad():
                    
                    # Calculate gradients with respect to the entires 
                    for i in range(self.tuning_length+1):
                        self.B_grads[i] = self.Bs[i].grad

                    # print(B_grads[tuning_length])
                    
                    # Calculate overall gradients 
                    ### version 1
                    # grad_B = self.B_grads[0]
                    ### version 2 / 3
                    # grad_B = torch.mean(torch.stack(self.B_grads), 0)
                    ### version 4
                    # # # # bias > 1 => favor recent
                    # # # # bias < 1 => favor earlier
                    bias = 1.5
                    weighted_grads_B = [None] * (self.tuning_length+1)
                    for i in range(self.tuning_length+1):
                        weighted_grads_B[i] = np.power(bias, i) * self.B_grads[i]
                    grad_B = torch.mean(torch.stack(weighted_grads_B), dim=0)
                    
                    # print(f'grad_B: {grad_B}')
                    # print(f'grad_B: {torch.norm(grad_B, 1)}')
                    

                    # Update parameters in time step t-H with saved gradients 
                    upd_B = self.binder.update_binding_matrix_(self.Bs[0], grad_B, self.at_learning_rate, self.bm_momentum)

                    # Compare binding matrix to ideal matrix
                    c_bm = self.binder.scale_binding_matrix(upd_B, self.scale_mode, self.scale_combo)
                    if order is not None: 
                        c_bm = c_bm.gather(1, reorder.unsqueeze(0).expand(c_bm.shape))

                        
                    mat_loss = self.evaluator.FBE(c_bm, self.ideal_binding)
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
                    g_h = at_h.grad
                    g_c = at_c.grad

                    upd_h = self.at_states[0][0] - self.at_learning_rate_state * g_h
                    upd_c = self.at_states[0][1] - self.at_learning_rate_state * g_c

                    at_h.data = upd_h.clone().detach().requires_grad_()
                    at_c.data = upd_c.clone().detach().requires_grad_()

                    at_h.grad.data.zero_()
                    at_c.grad.data.zero_()

                    # state_optimizer.step()
                    # print(f'updated init_state: {init_state}')
                
                ## REORGANIZE FOR MULTIPLE CYCLES!!!!!!!!!!!!!

                # forward pass from t-H to t with new parameters 
                init_state = (at_h, at_c)
                state = (init_state[0], init_state[1])
                for i in range(self.tuning_length):

                    bm = self.binder.scale_binding_matrix(self.Bs[i], self.scale_mode, self.scale_combo)
                    x_B = self.binder.bind(self.at_inputs[i], bm)
                    x = self.preprocessor.convert_data_AT_to_LSTM(x_B)

                    # print(f'x_B :{x_B}')

                    state = (state[0] * state_scaler, state[1] * state_scaler)
                    self.at_predictions[i], state = self.core_model(x, state)
                    
                    # for last tuning cycle update initial state to track gradients 
                    if cycle==(self.tuning_cycles-1) and i==0: 
                        with torch.no_grad():
                                final_prediction = self.at_predictions[0].clone().detach()
                                
                        at_h = state[0].clone().detach().requires_grad_().to(self.device)
                        at_c = state[1].clone().detach().requires_grad_().to(self.device)
                        init_state = (at_h, at_c)
                        state = (init_state[0], init_state[1])

                    self.at_states[i+1] = state 

                # Update current binding
                bm = self.binder.scale_binding_matrix(self.Bs[-1], self.scale_mode, self.scale_combo)
                x_B = self.binder.bind(o, bm)
                x = self.preprocessor.convert_data_AT_to_LSTM(x_B)


            # END tuning cycle        

            ## Generate updated prediction 
            state = self.at_states[-1]
            state = (state[0] * state_scaler, state[1] * state_scaler)
            new_prediction, state = self.core_model(x, state)

            ## Reorganize storage variables
            # states
            self.at_states.append(state)
            self.at_states[0][0].requires_grad = False
            self.at_states[0][1].requires_grad = False
            self.at_states = self.at_states[1:]
            
            # observations
            self.at_inputs = torch.cat((self.at_inputs[1:], o.reshape(1, self.num_input_features, self.num_input_dimensions)), 0)
            
            # predictions
            at_final_predictions = torch.cat((at_final_predictions, final_prediction.reshape(1,self.input_per_frame)), 0)
            self.at_predictions = torch.cat((self.at_predictions[1:], new_prediction.reshape(1,self.input_per_frame)), 0)

        # END active tuning
        
        # store rest of predictions in at_final_predictions
        for i in range(self.tuning_length): 
            at_final_predictions = torch.cat((at_final_predictions, self.at_predictions[1].reshape(1,self.input_per_frame)), 0)

        # get final binding matrix
        final_binding_matrix = self.binder.scale_binding_matrix(self.Bs[-1].clone().detach(), self.scale_mode, self.scale_combo)
        # print(f'final binding matrix: {final_binding_matrix}')
        final_binding_entries = self.Bs[-1].clone().detach()
        # print(f'final binding entires: {final_binding_entries}')

        return at_final_predictions, final_binding_matrix, final_binding_entries




    ############################################################################
    ##########  EVALUATION #####################################################

    def get_result_history(
        self, 
        observations, 
        at_final_predictions):

        pred_errors = self.evaluator.prediction_errors(observations, 
                                                at_final_predictions, 
                                                self.mse)

        return [pred_errors, self.at_losses, self.bm_losses, self.bm_dets]

