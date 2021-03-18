# packet imports 
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import angle, append 
import torch
import copy
from torch import nn, autograd
from torch.autograd import Variable
from torch._C import device
import matplotlib.pyplot as plt
from torch.functional import norm

# class imports 
from BinAndPerspTaking.binding import Binder
from BinAndPerspTaking.binding_exmat import BinderExMat
from BinAndPerspTaking.perspective_taking import Perspective_Taker
from CoreLSTM.core_lstm import CORE_NET
from Data_Compiler.data_preparation import Preprocessor
from BAPTAT_evaluation import BAPTAT_evaluator



class SEP_TRANSLATION():

    def __init__(self):
        ## General parameters 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autograd.set_detect_anomaly(True)

        torch.set_printoptions(precision=8)

        ## Set default parameters
        ##   -> Can be changed during experiments 

        self.grad_bias = 1.5

    
    ############################################################################
    ##########  PARAMETERS  ####################################################

    def set_weighted_gradient_bias(self, bias):
        # bias > 1 => favor recent
        # bias < 1 => favor earlier
        self.grad_bias = bias
        print(f'Reset bias for gradient weighting: {self.grad_bias}')


    def set_data_parameters_(self, num_frames, num_input_features, num_input_dimesions): 
        ## Define data parameters
        self.num_frames = num_frames
        self.num_input_features = num_input_features
        self.num_input_dimensions = num_input_dimesions
        self.input_per_frame = self.num_input_features * self.num_input_dimensions

        self.perspective_taker = Perspective_Taker(self.num_input_features, self.num_input_dimensions,
                                                   rotation_gradient_init=True, translation_gradient_init=True)
        self.preprocessor = Preprocessor(self.num_input_features, self.num_input_dimensions)
        self.evaluator = BAPTAT_evaluator(self.num_frames, self.num_input_features, self.preprocessor)
        
    
    def set_tuning_parameters_(self, tuning_length, num_tuning_cycles, loss_function, at_learning_rate_translation, at_learning_rate_state, at_momentum_translation): 
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
        self.at_learning_rate = at_learning_rate_translation
        self.at_learning_rate_state = at_learning_rate_state
        self.c_momentum = at_momentum_translation
        self.at_loss_function = self.mse

        print('Parameters set.')


    def init_model_(self, model_path): 
        ## Load model
        self.core_model = CORE_NET()
        self.core_model.load_state_dict(torch.load(model_path))
        self.core_model.eval()


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

        # translation
        self.Cs = []
        self.C_grads = [None] * (self.tuning_length+1)
        self.C_upd = [None] * (self.tuning_length+1)
        self.c_losses = []

    
    def set_comparison_values(self, ideal_translation):
        self.ideal_translation = ideal_translation


    ############################################################################
    ##########  INFERENCE  #####################################################
    
    def run_inference(self, observations, grad_calculation):

        at_final_predictions = torch.tensor([]).to(self.device)

        tb = self.perspective_taker.init_translation_bias_()
        print(tb)

        for i in range(self.tuning_length+1):
            transba = tb.clone()
            transba.requires_grad = True
            self.Cs.append(transba)


        ## Core state
        # define scaler
        state_scaler = 0.95

        # init state
        at_h = torch.zeros(1, self.core_model.hidden_size).requires_grad_().to(self.device)
        at_c = torch.zeros(1, self.core_model.hidden_size).requires_grad_().to(self.device)

        init_state = (at_h, at_c)
        state = (init_state[0], init_state[1])


        ############################################################################
        ##########  FORWARD PASS  ##################################################

        for i in range(self.tuning_length):
            o = observations[self.obs_count]
            self.at_inputs = torch.cat((self.at_inputs, o.reshape(1, self.num_input_features, self.num_input_dimensions)), 0)
            self.obs_count += 1

            x_C = self.perspective_taker.translate(o, self.Cs[i])
            x = self.preprocessor.convert_data_AT_to_LSTM(x_C)

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

            x_C = self.perspective_taker.translate(o, self.Cs[-1])
            x = self.preprocessor.convert_data_AT_to_LSTM(x_C)

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
                loss = self.at_loss(p, x[0])

                self.at_losses.append(loss)
                print(f'frame: {self.obs_count} cycle: {cycle} loss: {loss}')

                # Propagate error back through tuning horizon 
                loss.backward(retain_graph = True)

                # Update parameters 
                with torch.no_grad():
                    
                    ## Get gradients 
                    for i in range(self.tuning_length+1):
                        # save grads for all parameters in all time steps of tuning horizon 
                        self.C_grads[i] = self.Cs[i].grad

                    # print(self.C_grads[self.tuning_length])
                    
                    # Calculate overall gradients 
                    if grad_calculation == 'lastOfTunHor':
                        ### version 1
                        grad_C = self.C_grads[0]
                    elif grad_calculation == 'meanOfTunHor':
                        ### version 2 / 3
                        grad_C = torch.mean(torch.stack(self.C_grads), dim=0)
                    elif grad_calculation == 'weightedInTunHor':
                        ### version 4
                        weighted_grads_C = [None] * (self.tuning_length+1)
                        for i in range(self.tuning_length+1):
                            weighted_grads_C[i] = np.power(self.grad_bias, i) * self.C_grads[i]
                        grad_C = torch.mean(torch.stack(weighted_grads_C), dim=0)
                    
                    # print(f'grad_C: {grad_C}')

                    # Update parameters in time step t-H with saved gradients 
                    upd_C = self.perspective_taker.update_translation_bias_(self.Cs[0], grad_C, self.at_learning_rate, self.c_momentum)
                    
                    # print(upd_C)
                    # Compare translation bias to ideal bias
                    trans_loss = self.mse(self.ideal_translation, upd_C)
                    self.c_losses.append(trans_loss)
                    print(f'loss of translation bias (MSE): {trans_loss}')
                    
                    # Zero out gradients for all parameters in all time steps of tuning horizon
                    for i in range(self.tuning_length+1):
                        self.Cs[i].requires_grad = False
                        self.Cs[i].grad.data.zero_()
                    
                    # Update all parameters for all time steps 
                    for i in range(self.tuning_length+1):
                        translation = upd_C.clone()
                        translation.requires_grad_()
                        self.Cs[i] = translation

                    # print(self.Cs[0])

                    # Initial state
                    g_h = at_h.grad
                    g_c = at_c.grad

                    upd_h = init_state[0] - self.at_learning_rate_state * g_h
                    upd_c = init_state[1] - self.at_learning_rate_state * g_c

                    at_h.data = upd_h.clone().detach().requires_grad_()
                    at_c.data = upd_c.clone().detach().requires_grad_()

                    at_h.grad.data.zero_()
                    at_c.grad.data.zero_()

                    # state_optimizer.step()
                    # print(f'updated init_state: {init_state}')
                
                ## REORGANIZE FOR MULTIPLE CYCLES!!!!!!!!!!!!!

                # forward pass from t-H to t with new parameters 
                # Update init state???
                init_state = (at_h, at_c)
                state = (init_state[0], init_state[1])
                for i in range(self.tuning_length):

                    x_C = self.perspective_taker.translate(self.at_inputs[i], self.Cs[i])
                    x = self.preprocessor.convert_data_AT_to_LSTM(x_C)

                    state = (state[0] * state_scaler, state[1] * state_scaler)
                    self.at_predictions[i], state = self.core_model(x, state)

                    # for last tuning cycle update initial state to track gradients 
                    if cycle==(self.tuning_cycles-1) and i==0: 
                        with torch.no_grad():
                            final_prediction = self.at_predictions[0].clone().detach()

                        at_h = state[0].clone().detach().requires_grad_()
                        at_c = state[1].clone().detach().requires_grad_()
                        init_state = (at_h, at_c)
                        state = (init_state[0], init_state[1])

                    self.at_states[i] = state 

                # Update current rotation
                x_C = self.perspective_taker.translate(o, self.Cs[-1]) 
                x = self.preprocessor.convert_data_AT_to_LSTM(x_C)

            # END tuning cycle        

            ## Generate updated prediction 
            state = self.at_states[-1]
            state = (state[0] * state_scaler, state[1] * state_scaler)
            new_prediction, state = self.core_model(x, state)

            ## Reorganize storage variables            
            # observations
            self.at_inputs = torch.cat((self.at_inputs[1:], o.reshape(1, self.num_input_features, self.num_input_dimensions)), 0)
            
            # predictions
            at_final_predictions = torch.cat((at_final_predictions, final_prediction.reshape(1,self.input_per_frame)), 0)
            self.at_predictions = torch.cat((self.at_predictions[1:], new_prediction.reshape(1,self.input_per_frame)), 0)

        # END active tuning
            
        # store rest of predictions in at_final_predictions
        for i in range(self.tuning_length): 
            at_final_predictions = torch.cat((at_final_predictions, self.at_predictions[1].reshape(1,self.input_per_frame)), 0)


        # get final translation bias
        final_translation_bias = self.Cs[0].clone().detach()
        print(f'final translation bias: {final_translation_bias}')


        return at_final_predictions, final_translation_bias


    ############################################################################
    ##########  EVALUATION #####################################################

    def get_result_history(
        self, 
        observations, 
        at_final_predictions):

        pred_errors = self.evaluator.prediction_errors(observations, 
                                                at_final_predictions, 
                                                self.mse)

        return [pred_errors, self.at_losses, self.c_losses]



