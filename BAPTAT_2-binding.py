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


## Define data parameters
num_frames = 300
num_input_features = 15
num_input_dimensions = 3
preprocessor = Preprocessor(num_input_features, num_input_dimensions)
evaluator = BAPTAT_evaluator(num_frames, preprocessor)
data_at_unlike_train = False ## Note: sample needs to be changed in the future

# data paths 
data_asf_path = 'Data_Compiler/S35T07.asf'
data_amc_path = 'Data_Compiler/S35T07.amc'


## Define model parameters 
model_path = 'CoreLSTM/models/LSTM_23.pt'


## Define tuning parameters 
tuning_length = 20      # length of tuning horizon 
tuning_cycles = 1       # number of tuning cycles in each iteration 
at_loss_function = nn.MSELoss()
# mse = nn.MSELoss()
# at_loss_function = lambda x,y: mse(x, y) * (num_input_dimensions * num_input_features)
at_learning_rate = 0.1
at_learning_rate_state = 0.001


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

# binding
bindSM = nn.Softmax(dim=0)  # columnwise
# binder = Binder(num_features=num_input_features, gradient_init=True)
binder = BinderExMat(num_features=num_input_features, gradient_init=True)
ideal_binding = torch.Tensor(np.identity(num_input_features))

Bs = []
B_grads = [None] * (tuning_length+1)
B_upd = [None] * (tuning_length+1)
bm_losses = []



############################################################################
##########  INITIALIZATIONS  ###############################################

## Load data
observations, feature_names = preprocessor.get_AT_data(data_asf_path, data_amc_path, num_frames)
    

## Load model
core_model = CORE_NET()
core_model.load_state_dict(torch.load(model_path))
core_model.eval()


## Binding matrices 
# Version 1: Init binding matrices
# bm = binder.init_binding_matrix_()

# for i in range(tuning_length+1):
#     binmat = copy.deepcopy(bm)
#     binmat.requires_grad = True
#     Bs.append(binmat)

# Version 2: Init binding entries 
be = binder.init_binding_entries_()
print(be)

for i in range(tuning_length+1):
    entries = []
    for j in range(num_input_features):
        row = []
        for k in range(num_input_features):
            entry = be[j][k].clone()
            entry.requires_grad_()
            row.append(entry)
        entries.append(row)
    Bs.append(entries)
    
print(f'BMs different in list: {Bs[0] is not Bs[1]}')


## Core state
at_h = torch.zeros(core_model.hidden_num, 1, core_model.hidden_size).requires_grad_()
at_c = torch.zeros(core_model.hidden_num, 1, core_model.hidden_size).requires_grad_()
init_state = (at_h, at_c)
at_states.append(init_state)
state = (init_state[0], init_state[1])


############################################################################
##########  FORWARD PASS  ##################################################

for i in range(tuning_length):
    o = observations[obs_count]
    at_inputs = torch.cat((at_inputs, o.reshape(1, num_input_features, num_input_dimensions)), 0)
    obs_count += 1

    # Version 1
    # x_B = binder.bind(o, bindSM(Bs[i]))
    # Version 2
    bm = binder.compute_binding_matrix(Bs[i], bindSM)
    x_B = binder.bind(o, bm)
    x = preprocessor.convert_data_AT_to_LSTM(x_B)

    # with torch.no_grad():
    new_prediction, state = core_model(x, state)  
    at_states.append(state)
    at_predictions = torch.cat((at_predictions, new_prediction.reshape(1,45)), 0)
    ##### outside torch.no_grad????????? -> otherwise no gradients 


############################################################################
##########  ACTIVE TUNING ##################################################

while obs_count < num_frames:
    # TODO folgendes evtl in function auslagern
    o = observations[obs_count]
    obs_count += 1

    # Version 1
    # x_B = binder.bind(o, bindSM(Bs[-1]))
    # Version 2
    bm = binder.compute_binding_matrix(Bs[-1], bindSM)
    x_B = binder.bind(o, bm)
    
    x = preprocessor.convert_data_AT_to_LSTM(x_B)

    ## Generate current prediction 
    with torch.no_grad():
        new_prediction, state_new = core_model(x, at_states[-1])

    ## For #tuning_cycles 
    for cycle in range(tuning_cycles):
        print('----------------------------------------------')

        # Get prediction
        p = at_predictions[-1]

        # Calculate error 
        loss = at_loss_function(p, x[0]) 
        at_losses.append(loss)
        print(f'frame: {obs_count} cycle: {cycle} loss: {loss}')

        # Propagate error back through tuning horizon 
        loss.backward(retain_graph = True)

        # Update parameters 
        with torch.no_grad():
            
            ## Binding Matrix
            # Version 1
            # for i in range(tuning_length+1):
            #     # save grads for all parameters in all time steps of tuning horizon 
            #     B_grads[i] = Bs[i].grad
            #     # print(Bs[i].grad)

            # Version 2
            for i in range(tuning_length+1):
                grad = []
                for j in range(num_input_features):
                    row = []
                    for k in range(num_input_features):
                        row.append(Bs[i][j][k].grad)
                    grad.append(torch.stack(row))
                B_grads[i] = torch.stack(grad)

            # print(B_grads[tuning_length])

            
            # Calculate overall gradients 
            ### version 1
            # grad_B = B_grads[0]
            ### version 2 / 3
            # grad_B = torch.mean(torch.stack(B_grads))
            ### version 4
            bias = 8
            # # bias > 1 => favor recent
            # # bias < 1 => favor earlier
            weighted_grads_B = [None] * (tuning_length+1)
            for i in range(tuning_length+1):
                weighted_grads_B[i] = np.power(bias, i) * B_grads[i]
            grad_B = torch.mean(torch.stack(weighted_grads_B), dim=1)
            
            # print(f'grad_B: {grad_B}')

            # Update parameters in time step t-H with saved gradients 
            # Version 1
            # upd_B = binder.update_binding_matrix_(Bs[0], grad_B, at_learning_rate)
            # for i in range(tuning_length+1):
            #     B_upd[i] = binder.update_binding_matrix_(Bs[i], grad_B, at_learning_rate)
 
            # Version 2
            upd_B = binder.update_binding_entries_(Bs[0], grad_B, at_learning_rate)

            # Compare binding matrix to ideal matrix
            mat_loss = at_loss_function(ideal_binding, binder.compute_binding_matrix(upd_B, bindSM)) 
            # mat_loss = at_loss_function(ideal_binding, B_upd[0])
            bm_losses.append(mat_loss)
            print(f'loss of binding matrix: {mat_loss}')
            
            # Zero out gradients for all parameters in all time steps of tuning horizon
            for i in range(tuning_length+1):
                for j in range(num_input_features):
                    for k in range(num_input_features):
                        Bs[i][j][k].requires_grad = False
                        Bs[i][j][k].grad.data.zero_()

            # Update all parameters for all time steps 
            # Version 1
            # for i in range(tuning_length+1):
            #     Bs[i].data = upd_B.clone().data
            #     # Bs[i].data = B_upd[i].clone().data

            # Version 2
            for i in range(tuning_length):
                entries = []
                for j in range(num_input_features):
                    row = []
                    for k in range(num_input_features):
                        entry = upd_B[j][k].clone()
                        entry.requires_grad_()
                        row.append(entry)
                    entries.append(row)
                Bs[i] = entries

            
            # print(Bs[0])


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
        # Update init state???
        init_state = (at_h, at_c)
        state = (init_state[0], init_state[1])
        for i in range(tuning_length):

            # Version 1
            # x_B = binder.bind(o, bindSM(Bs[i]))
            # Version 2
            bm = binder.compute_binding_matrix(Bs[i], bindSM)
            x_B = binder.bind(o, bm)
            x = preprocessor.convert_data_AT_to_LSTM(x_B)

            # print(f'x_B :{x_B}')

            at_predictions[i], state = core_model(x, state)
            # for last tuning cycle update initial state to track gradients 
            if cycle==(tuning_cycles-1) and i==0: 
                at_h = state[0].clone().detach().requires_grad_()
                at_c = state[1].clone().detach().requires_grad_()
                init_state = (at_h, at_c)
                state = (init_state[0], init_state[1])

            at_states[i+1] = state 

        # Update current binding
        # Version 1
        # x_B = binder.bind(o, bindSM(Bs[-1]))
        # Version 2
        bm = binder.compute_binding_matrix(Bs[-1], bindSM)
        x_B = binder.bind(o, bm)
        x = preprocessor.convert_data_AT_to_LSTM(x_B)


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

# get final binding matrix
final_binding_matrix = binder.compute_binding_matrix(Bs[-1], bindSM)
print(f'final binding matrix: {final_binding_matrix}')


############################################################################
##########  EVALUATION #####################################################
pred_errors = evaluator.prediction_errors(observations, 
                                          at_final_predictions, 
                                          at_loss_function)

evaluator.plot_at_losses(at_losses)
evaluator.plot_at_losses(bm_losses)

evaluator.plot_binding_matrix(final_binding_matrix, feature_names)

# evaluator.help_visualize_devel(observations, at_final_predictions)
