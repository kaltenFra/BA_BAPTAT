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
num_frames = 50
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
smL1Loss = nn.SmoothL1Loss()
# smL1Loss = nn.SmoothL1Loss(beta=0.8)
# smL1Loss = nn.SmoothL1Loss(reduction='sum', beta=0.8)
l2Loss = lambda x,y: mse(x, y) * (num_input_dimensions * num_input_features)

# define learning parameters 
lstm_loss_function = mse
at_learning_rate = 1
# TODO try different learning rates 
at_learning_rate_state = 0.0
# loss_scale_factor = 0.6

bm_momentum = 0.0
c_momentum = 0.0
r_momentum = 0.0

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
binder = BinderExMat(num_features=num_input_features, gradient_init=True)
ideal_binding = torch.Tensor(np.identity(num_input_features))

Bs = []
B_grads = [None] * (tuning_length+1)
B_upd = [None] * (tuning_length+1)
bm_losses = []
bm_dets = []

# translation
perspective_taker = Perspective_Taker(num_input_features, num_input_dimensions,
                    rotation_gradient_init=True, translation_gradient_init=True)
ideal_trans = torch.zeros(3)

Cs = []
C_grads = [None] * (tuning_length+1)
C_upd = [None] * (tuning_length+1)
c_losses = []
c_norms = []

# rotation
ideal_quat = torch.zeros(1,num_input_dimensions)
ideal_rotation = torch.Tensor(np.identity(num_input_dimensions))

Rs = []
R_grads = [None] * (tuning_length+1)
R_upd = [None] * (tuning_length+1)
rm_losses = []
ra_losses = []


############################################################################
##########  INITIALIZATIONS  ###############################################

## Load data
observations, feature_names = preprocessor.get_AT_data(data_asf_path, data_amc_path, num_frames)
    

## Load model
core_model = CORE_NET()
core_model.load_state_dict(torch.load(model_path))
core_model.eval()


## Binding matrices 
# Init binding entries 
bm = binder.init_binding_matrix_det_()
# bm = binder.init_binding_matrix_rand_()
print(bm)

for i in range(tuning_length+1):
    matrix = bm.clone()
    matrix.requires_grad_()
    Bs.append(matrix)
    
print(f'BMs different in list: {Bs[0] is not Bs[1]}')


## Translation biases 
tb = perspective_taker.init_translation_bias_()

for i in range(tuning_length+1):
    # transba = copy.deepcopy(tb)
    transba = tb.clone()
    transba.requires_grad = True
    Cs.append(transba)

print(f'BMs different in list: {Cs[0] is not Cs[1]}')


## Rotation quaternion 
rq = perspective_taker.init_quaternion()
print(rq)

for i in range(tuning_length+1):
    quat = rq.clone()
    quat.requires_grad_()
    Rs.append(quat)

print(Rs[0])
print(f'RAs different in list: {Rs[0] is not Rs[1]}')


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

    # compute matrices
    bm = binder.scale_binding_matrix(Bs[i])
    
    # perform translation, binding and rotation
    x_C = perspective_taker.translate(o, Cs[i])
    x_B = binder.bind(x_C, bm)
    x_R = perspective_taker.qrotate(x_B, Rs[i])
    x = preprocessor.convert_data_AT_to_LSTM(x_R)

    # make prediction
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

    # compute matrices
    bm = binder.scale_binding_matrix(Bs[-1])
    
    # perform translation, binding and rotation
    x_C = perspective_taker.translate(o, Cs[-1])
    x_B = binder.bind(x_C, bm)
    x_R = perspective_taker.qrotate(x_B, Rs[-1])
    x = preprocessor.convert_data_AT_to_LSTM(x_R)

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
        # loss_scale = torch.square(torch.mean(torch.norm(bm.clone().detach(), dim=1, keepdim=True)) -1.) ##COPY?????
        # print(f'loss scale: {loss_scale}')
        # l1scale = loss_scale_factor * loss_scale
        # l2scale = loss_scale_factor / loss_scale

        # loss = loss_scale_factor * loss_scale * l2Loss(p,x[0]) + mse(p,x[0])
        # loss = loss_scale_factor * l2Loss(p,x[0]) + mse(p,x[0])
        # loss = mse(p,x[0])
        loss = smL1Loss(p, x[0])

        at_losses.append(loss)
        print(f'frame: {obs_count} cycle: {cycle} loss: {loss}')

        # Propagate error back through tuning horizon 
        loss.backward(retain_graph = True)

        # Update parameters 
        with torch.no_grad():
            
            #################### BINDING ####################
            # Calculate gradients with respect to the entires 
            for i in range(tuning_length+1):
                B_grads[i] = Bs[i].grad

            # print(B_grads[tuning_length])
            
            # Calculate overall gradients 
            ### version 1
            # grad_B = B_grads[0]
            ### version 2 / 3
            # grad_B = torch.mean(torch.stack(B_grads), 0)
            ### version 4
            # # # # bias > 1 => favor recent
            # # # # bias < 1 => favor earlier
            bias = 1.5
            weighted_grads_B = [None] * (tuning_length+1)
            for i in range(tuning_length+1):
                weighted_grads_B[i] = np.power(bias, i) * B_grads[i]
            grad_B = torch.mean(torch.stack(weighted_grads_B), dim=0)
            
            # print(f'grad_B: {grad_B}')
            # print(f'grad_B: {torch.norm(grad_B, 1)}')
            

            # Update parameters in time step t-H with saved gradients 
            upd_B = binder.update_binding_matrix_(Bs[0], grad_B, at_learning_rate, bm_momentum)

            # Compare binding matrix to ideal matrix
            c_bm = binder.scale_binding_matrix(upd_B)
            mat_loss = evaluator.FBE(c_bm, ideal_binding)
            bm_losses.append(mat_loss)
            print(f'loss of binding matrix (FBE): {mat_loss}')

            # Compute determinante of binding matrix
            det = torch.det(c_bm)
            bm_dets.append(det)
            print(f'determinante of binding matrix: {det}')
            
            # Zero out gradients for all parameters in all time steps of tuning horizon
            for i in range(tuning_length+1):
                Bs[i].requires_grad = False
                Bs[i].grad.data.zero_()

            # Update all parameters for all time steps 
            for i in range(tuning_length+1):
                Bs[i].data = upd_B.clone().data
                Bs[i].requires_grad = True
            
            # print(Bs[0])

            #################### TRANSLATION ####################
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
            
            # print(f'grad_C: {grad_C}')

            # Update parameters in time step t-H with saved gradients 
            upd_C = perspective_taker.update_translation_bias_(Cs[0], grad_C, at_learning_rate, c_momentum)
            # for i in range(tuning_length+1):
            #     C_upd[i] = perspective_taker.update_translation_bias_(Cs[i], grad_C, at_learning_rate)
            # print(upd_C)

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

            #################### ROTATION ####################

            ## Rotation Matrix
            for i in range(tuning_length+1):
                # save grads for all parameters in all time steps of tuning horizon
                R_grads[i] = Rs[i].grad
            # print(R_grads[tuning_length])
            
            # Calculate overall gradients 
            ### version 1
            # grad_R = R_grads[0]
            ### version 2 / 3
            grad_R = torch.mean(torch.stack(R_grads), dim=0)
            ### version 4
            # # bias > 1 => favor recent
            # # bias < 1 => favor earlier
            # bias = 1.5
            # weighted_grads_R = [None] * (tuning_length+1)
            # for i in range(tuning_length+1):
            #     weighted_grads_R[i] = np.power(bias, i) * R_grads[i]
            # grad_R = torch.mean(torch.stack(weighted_grads_R), dim=0)
            
            # print(f'grad_R: {grad_R}')

            # Update parameters in time step t-H with saved gradients 
            upd_R = perspective_taker.update_quaternion(Rs[0], grad_R, at_learning_rate, r_momentum)
            print(f'updated quaternion: {upd_R}')

            # Compare to ideal rotation
            rotmat = perspective_taker.quaternion2rotmat(upd_R)
            mat_loss = mse(ideal_rotation, rotmat)
            print(f'loss of rotation matrix: {mat_loss}')
            rm_losses.append(mat_loss)


            # Zero out gradients for all parameters in all time steps of tuning horizon
            for i in range(tuning_length+1):
                Rs[i].requires_grad = False
                Rs[i].grad.data.zero_()

            # Update all parameters for all time steps 
            for i in range(tuning_length+1):
                quat = upd_R.clone()
                quat.requires_grad_()
                Rs[i] = quat
            # print(Rs[0])

            #################### CELL STATE ####################

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

            # compute matrices
            bm = binder.scale_binding_matrix(Bs[i])
            
            # perform translation, binding and rotation
            x_C = perspective_taker.translate(o, Cs[i])
            x_B = binder.bind(x_C, bm)
            x_R = perspective_taker.qrotate(x_B, Rs[i])
            x = preprocessor.convert_data_AT_to_LSTM(x_R)

            # make prediction
            state = (state[0] * state_scaler, state[1] * state_scaler)
            at_predictions[i], state = core_model(x, state)
            
            # for last tuning cycle update initial state to track gradients 
            if cycle==(tuning_cycles-1) and i==0: 
                at_h = state[0].clone().detach().requires_grad_()
                at_c = state[1].clone().detach().requires_grad_()
                init_state = (at_h, at_c)
                state = (init_state[0], init_state[1])

            at_states[i+1] = state 

        # Update current parameters
        # compute matrices
        bm = binder.scale_binding_matrix(Bs[-1])
        
        # perform translation, binding and rotation
        x_C = perspective_taker.translate(o, Cs[-1])
        x_B = binder.bind(x_C, bm)
        x_R = perspective_taker.qrotate(x_B, Rs[-1])
        x = preprocessor.convert_data_AT_to_LSTM(x_R)


    # END tuning cycle        

    ## Generate updated prediction 
    state = at_states[-1]
    state = (state[0] * state_scaler, state[1] * state_scaler)
    new_prediction, state = core_model(x, state)

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
final_binding_matrix = binder.scale_binding_matrix(Bs[-1])
print(f'final binding matrix: {final_binding_matrix}')
final_binding_entires = torch.tensor(Bs[-1])
print(f'final binding entires: {final_binding_entires}')

# get final translation bias
final_translation_bias = Cs[0]
print(f'final translation bias: {final_translation_bias}')


############################################################################
##########  EVALUATION #####################################################
pred_errors = evaluator.prediction_errors(observations, 
                                          at_final_predictions, 
                                          lstm_loss_function)

evaluator.plot_at_losses(at_losses, 'History of overall losses during active tuning')
evaluator.plot_at_losses(bm_losses, 'History of binding matrix loss (FBE)')
evaluator.plot_at_losses(bm_dets, 'History of binding matrix determinante')
evaluator.plot_at_losses(c_losses,'History of translation bias loss (MSE)')
evaluator.plot_at_losses(c_norms,'History of translation bias norms')
evaluator.plot_at_losses(rm_losses,'History of rotaion matrix loss (MSE)')

evaluator.plot_binding_matrix(
    final_binding_matrix, 
    feature_names, 
    'Binding matrix showing relative contribution of observed feature to input feature'
)
evaluator.plot_binding_matrix(
    final_binding_entires, 
    feature_names, 
    'Binding matrix entries showing contribution of observed feature to input feature'
)

# evaluator.help_visualize_devel(observations, at_final_predictions)
