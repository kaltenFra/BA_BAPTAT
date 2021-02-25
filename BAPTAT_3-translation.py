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

############################################################################
##########  PARAMETERS  ####################################################

## General parameters 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autograd.set_detect_anomaly(True)

torch.set_printoptions(precision=8)

## Define data parameters
num_frames = 30
num_input_features = 15
num_input_dimensions = 3
preprocessor = Preprocessor(num_input_features, num_input_dimensions)
evaluator = BAPTAT_evaluator(num_frames, num_input_features, preprocessor)
data_at_unlike_train = False ## Note: sample needs to be changed in the future

# data paths 
data_asf_path = 'Data_Compiler/S35T07.asf'
data_amc_path = 'Data_Compiler/S35T07.amc'


## Define model parameters 
model_path = 'CoreLSTM/models/LSTM_46_cell.pt'


## Define tuning parameters 
tuning_length = 10      # length of tuning horizon 
tuning_cycles = 3       # number of tuning cycles in each iteration 

# possible loss functions
mse = nn.MSELoss()
l1Loss = nn.L1Loss()
l2Loss = lambda x,y: mse(x, y) * (num_input_dimensions * num_input_features)

# define learning parameters 
at_loss_function = l1Loss
at_learning_rate = 1
at_learning_rate_state = 0.001
c_momentum = 0.0

## Define tuning variables
# general
obs_count = 0
at_inputs = torch.tensor([])
at_predictions = torch.tensor([])
at_final_predictions = torch.tensor([])
at_losses = []

# state 
at_states = []
# state_optimizer = torch.optim.Adam(init_state, at_learning_rate)

# translation
perspective_taker = Perspective_Taker(num_input_features, num_input_dimensions,
                    rotation_gradient_init=True, translation_gradient_init=True)
ideal_trans = torch.zeros(3)

Cs = []
C_grads = [None] * (tuning_length+1)
C_upd = [None] * (tuning_length+1)
c_losses = []
c_norms = []
c_norm = 1




############################################################################
##########  INITIALIZATIONS  ###############################################

## Load data
observations, feature_names = preprocessor.get_AT_data(data_asf_path, data_amc_path, num_frames)
    

## Load model
core_model = CORE_NET()
core_model.load_state_dict(torch.load(model_path))
core_model.eval()


## Translation biases 
tb = perspective_taker.init_translation_bias_()
# tb = torch.Tensor([0.5,0.5,0.5])
# tb = torch.Tensor([-0.5,0.5,0.5])
# tb = torch.Tensor([-0.5,-0.5,0.5])
# tb = torch.Tensor([-0.5,-0.5,-0.5])
print(tb)

for i in range(tuning_length+1):
    # transba = copy.deepcopy(tb)
    transba = tb.clone()
    transba.requires_grad = True
    Cs.append(transba)

print(f'CBs different in list: {Cs[0] is not Cs[1]}')


## Core state
# define scaler
state_scaler = 0.95

# init state
at_h = torch.zeros(1, core_model.hidden_size).requires_grad_()
at_c = torch.zeros(1, core_model.hidden_size).requires_grad_()

init_state = (at_h, at_c)
at_states.append(init_state)
state = (init_state[0], init_state[1])


############################################################################
##########  FORWARD PASS  ##################################################

for i in range(tuning_length):
    o = observations[obs_count]
    at_inputs = torch.cat((at_inputs, o.reshape(1, num_input_features, num_input_dimensions)), 0)
    obs_count += 1

    # x_C = perspective_taker.translate(o, Cs[i]/c_norm)
    x_C = perspective_taker.translate(o, Cs[i])

    x = preprocessor.convert_data_AT_to_LSTM(x_C)

    state = (state[0] * state_scaler, state[1] * state_scaler)
    new_prediction, state = core_model(x, state)  
    at_states.append(state)
    at_predictions = torch.cat((at_predictions, new_prediction.reshape(1,45)), 0)

############################################################################
##########  ACTIVE TUNING ##################################################

while obs_count < num_frames:
    # TODO folgendes evtl in function auslagern
    o = observations[obs_count]
    obs_count += 1

    # x_C = perspective_taker.translate(o, Cs[-1]/c_norm)
    x_C = perspective_taker.translate(o, Cs[-1])

    x = preprocessor.convert_data_AT_to_LSTM(x_C)

    ## Generate current prediction 
    with torch.no_grad():
        state = (state[0] * state_scaler, state[1] * state_scaler)
        new_prediction, state_new = core_model(x, at_states[-1])

    ## For #tuning_cycles 
    for cycle in range(tuning_cycles):
        print('----------------------------------------------')

        # Get prediction
        p = at_predictions[-1]

        # Calculate error 
        loss_scale_factor = 0.1
        # loss = mse(p,x[0])
        loss = loss_scale_factor * l2Loss(p,x[0]) + mse(p,x[0])
        # loss = loss_scale_factor * l1Loss(p,x[0]) + mse(p,x[0])

        at_losses.append(loss)
        print(f'frame: {obs_count} cycle: {cycle} loss: {loss}')

        # Propagate error back through tuning horizon 
        loss.backward(retain_graph = True)

        # Update parameters 
        with torch.no_grad():
            ## Normalization factor
            c_norm = loss.detach()
            
            # Get gradients 
            for i in range(tuning_length+1):
                # save grads for all parameters in all time steps of tuning horizon 
                C_grads[i] = Cs[i].grad
                # print(Cs[i].grad)
            # print(C_grads[tuning_length])
            
            # Calculate overall gradients 
            ### version 1
            # grad_C = C_grads[0]
            ### version 2 / 3
            grad_C = torch.mean(torch.stack(C_grads),0)
            ### version 4
            # bias = 10
            # # bias > 1 => favor recent
            # # bias < 1 => favor earlier
            # weighted_grads_C = [None] * (tuning_length+1)
            # for i in range(tuning_length+1):
            #     weighted_grads_C[i] = np.power(bias, i) * C_grads[i]
            # grad_C = torch.mean(torch.stack(weighted_grads_C))
            
            print(f'grad_C: {grad_C}')

            # Update parameters in time step t-H with saved gradients 
            upd_C = perspective_taker.update_translation_bias_(Cs[0], grad_C, at_learning_rate, c_momentum)
            # for i in range(tuning_length+1):
            #     C_upd[i] = perspective_taker.update_translation_bias_(Cs[i], grad_C, at_learning_rate)
            print(upd_C)

            # Compare translation bias to ideal bias
            trans_loss = mse(ideal_trans, upd_C)
            c_losses.append(trans_loss)
            print(f'loss of translation bias (MSE): {trans_loss}')

            # Compute norm of bias
            trans_norm = torch.norm(upd_C)
            c_norms.append(trans_loss)
            print(f'norm of translation bias: {trans_norm}')

            # Zero out gradients for all parameters in all time steps of tuning horizon
            for i in range(tuning_length+1):
                Cs[i].grad.data.zero_()
            
            # Update all parameters for all time steps 
            for i in range(tuning_length+1):
                Cs[i].data = upd_C.clone().data

            # print(Cs[0])


            # Initial state
            g_h = at_h.grad
            g_c = at_c.grad

            upd_h = at_states[0][0] - at_learning_rate_state * g_h
            upd_c = at_states[0][1] - at_learning_rate_state * g_c

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
        for i in range(tuning_length):

            # x_C = perspective_taker.translate(at_inputs[i], Cs[i]/c_norm)
            x_C = perspective_taker.translate(at_inputs[i], Cs[i])
            x = preprocessor.convert_data_AT_to_LSTM(x_C)

            # print(f'x_B :{x_B}')

            state = (state[0] * state_scaler, state[1] * state_scaler)
            at_predictions[i], state = core_model(x, state)

            # for last tuning cycle update initial state to track gradients 
            if cycle==(tuning_cycles-1) and i==0: 
                at_h = state[0].clone().detach().requires_grad_()
                at_c = state[1].clone().detach().requires_grad_()
                init_state = (at_h, at_c)
                state = (init_state[0], init_state[1])

            at_states[i+1] = state 

        # Update current binding
        # x_C = perspective_taker.translate(o, Cs[-1]/c_norm) 
        x_C = perspective_taker.translate(o, Cs[-1]) 
        x = preprocessor.convert_data_AT_to_LSTM(x_C)

    # END tuning cycle        

    ## Generate updated prediction 
    new_prediction, state = core_model(x, at_states[-1])

    ## Reorganize storage variables
    # states
    at_states.append(state)
    at_states[0][0].requires_grad = False
    at_states[0][1].requires_grad = False
    at_states = at_states[1:]
    
    # observations
    at_inputs = torch.cat((at_inputs[1:], o.reshape(1, num_input_features, num_input_dimensions)), 0)
    
    # predictions
    at_final_predictions = torch.cat((at_final_predictions, at_predictions[0].detach().reshape(1,45)), 0)
    at_predictions = torch.cat((at_predictions[1:], new_prediction.reshape(1,45)), 0)

# END active tuning
    
# store rest of predictions in at_final_predictions
for i in range(tuning_length): 
    at_final_predictions = torch.cat((at_final_predictions, at_predictions[1].reshape(1,45)), 0)

# get final translation bias
final_translation_bias = Cs[0]
print(f'final translation bias: {final_translation_bias}')


############################################################################
##########  EVALUATION #####################################################
pred_errors = evaluator.prediction_errors(observations, 
                                          at_final_predictions, 
                                          at_loss_function)

evaluator.plot_at_losses(at_losses, 'History of overall losses during active tuning')
evaluator.plot_at_losses(c_losses,'History of translation bias loss (MSE)')
evaluator.plot_at_losses(c_norms,'History of translation bias norms')

# evaluator.help_visualize_devel(observations, at_final_predictions)
