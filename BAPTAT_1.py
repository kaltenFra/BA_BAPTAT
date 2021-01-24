# packet imports 
import numpy as np 
import torch
from torch import nn, autograd
from torch.autograd import Variable
from torch._C import device
import matplotlib.pyplot as plt

# class imports 
from BinAndPerspTaking.binding import Binder
from BinAndPerspTaking.perspective_taking import Perspective_Taker
from CoreLSTM.core_lstm import CORE_NET
from Data_Compiler.data_preparation import Preprocessor
from BAPTAT_evaluation import BAPTAT_evaluator

## General parameters 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autograd.set_detect_anomaly(True)

## Define tuning parameters 
tuning_length = 10      # length of tuning horizon 
tuning_cycles = 3       # number of tuning cycles in each iteration 
at_loss_function = nn.MSELoss()

at_final_predictions = torch.tensor([])
B_grads = [None] * tuning_length
C_grads = [None] * tuning_length
R_grads = [None] * tuning_length

at_learning_rate = 0.001


## Define data parameters
data_at_unlike_train = True
num_frames = 90
num_input_features = 15
num_input_dimensions = 3
preprocessor = Preprocessor(num_input_features, num_input_dimensions)
evaluator = BAPTAT_evaluator(num_frames, preprocessor)

# data paths 
data_asf_path = 'Data_Compiler/S35T07.asf'
data_amc_path = 'Data_Compiler/S35T07.amc'

####################### Note: sample needs to be changed in the future

## Define model parameters 
model_path = 'CoreLSTM/models/LSTM_2.pt'

## Define Binding and Perspektive Taking parameters 
# Binding
binder = Binder(num_features=num_input_features, gradient_init=True)

# Perspective Taking
perspective_taker = Perspective_Taker(alpha_init=0.0, beta_init=0.0, gamma_init=0.0, 
                    rotation_gradient_init=True, translation_gradient_init=True)


## Create list of parameters for gradient descent
Bs = []
Cs = []
Rs = []
################################################### appending separate matrices or only references?
###################### maybe tensors not lists? -> backprop possible 
for i in range(tuning_length):
    Bs.append(binder.binding_matrix_())
    Cs.append(perspective_taker.translation_bias_())
    Rs.append(perspective_taker.rotation_matrix_())

# print(Bs[0])
# print(Cs[0])
# print(Rs[0])

## Load data
observations, feature_names = preprocessor.get_AT_data(data_asf_path, data_amc_path, num_frames)

## Change data according to given parameters
if (data_at_unlike_train):
    ## translate data 
    init_translation_bias = torch.Tensor([3.0, 3.0, 3.0])
    print(f'initial translation bias:\n {init_translation_bias}')
    for i in range(num_frames):
        observations[i] = perspective_taker.translate(observations[i], init_translation_bias)

    ## rotate data 
    # rotation angles 
    init_alpha = Variable(torch.tensor(0.1))
    init_beta = Variable(torch.tensor(0.1))
    init_gamma = Variable(torch.tensor(0.1))

    # dimensional rotation matrices 
    init_R_x = Variable(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, torch.cos(init_alpha), - torch.sin(init_alpha)], 
        [0.0, torch.sin(init_alpha), torch.cos(init_alpha)]])) 
    init_R_y = Variable(torch.tensor([
        [torch.cos(init_beta), 0.0, torch.sin(init_beta)], 
        [0.0,1.0,0.0],
        [- torch.sin(init_beta), 0.0, torch.cos(init_beta)]])) 
    init_R_z = Variable(torch.tensor([
        [torch.cos(init_gamma), - torch.sin(init_gamma), 0.0], 
        [torch.sin(init_gamma), torch.cos(init_gamma), 0.0], 
        [0.0,0.0,1.0]]))

    # rotation matrix
    init_rotation_matrix = Variable(torch.matmul(init_R_z, torch.matmul(init_R_y, init_R_x)))
    print(f'initial rotation matrix:\n {init_rotation_matrix}')
    for i in range(num_frames):
        observations[i] = perspective_taker.rotate(observations[i], init_rotation_matrix)

    ## rebind data
    # TODO
    
## Load model
core_model = CORE_NET()
core_model.load_state_dict(torch.load(model_path))
core_model.eval()

#### FORWARD PASS FOR ONE TUNING HORIZON --- NOT closed loop! preds based on observations
obs_count = 0
at_inputs = torch.tensor([])
at_predictions = torch.tensor([])

for i in range(tuning_length):
    o = observations[obs_count]
    obs_count += 1

    x_B = binder.bind(o, Bs[i])
    x_C = perspective_taker.translate(x_B, Cs[i])
    x_R = perspective_taker.rotate(x_C, Rs[i])

    x = preprocessor.convert_data_AT_to_LSTM(x_R)

    at_inputs = torch.cat((at_inputs, x), 0)
    
    with torch.no_grad():
        new_prediction, state = core_model(x)
        at_predictions = torch.cat((at_predictions, new_prediction.reshape(1,45)), 0)


#### ACTIVE TUNING PART 

while obs_count < num_frames:
    # TODO folgendes evtl in function auslagern
    o = observations[obs_count]
    obs_count += 1

    x_B = binder.bind(o, Bs[-1])
    x_C = perspective_taker.translate(x_B, Cs[-1])
    x_R = perspective_taker.rotate(x_C, Rs[-1])

    x = preprocessor.convert_data_AT_to_LSTM(x_R)

    ## Generate current prediction 
    with torch.no_grad():
        new_prediction, state = core_model(x)

    ## For #tuning_cycles 
    for cycle in range(tuning_cycles):
        print('----------------------------------------------')

        # get prediction
        p = at_predictions[-1]

        # calculate error 
        loss = at_loss_function(p, x)
        print(f'frame: {obs_count} cycle: {cycle} loss: {loss}')

        # propagate error back through tuning horizon 
        loss.backward(retain_graph=True)
        ############################# retain_graph = True ? 
        ############################# optimizer? Adam.. 

        # for i in TH
        for i in range(tuning_length):
            # save grads for all parameters in all time steps of tuning horizon 
            B_grads[i] = Bs[i].grad
            C_grads[i] = Cs[i].grad
            R_grads[i] = Rs[i].grad
            # Working? not all tensors.. maybe add Varables? --> added Variables  
        
        # calculate overall gradients 
        grad_B = torch.mean(torch.stack(B_grads))
        grad_C = torch.mean(torch.stack(C_grads))
        grad_R = torch.mean(torch.stack(R_grads))

        # update parameters in time step t-H with saved gradients 
        upd_B = binder.update_binding_matrix_(grad_B, at_learning_rate)
        upd_C = perspective_taker.update_translation_bias_(grad_C, at_learning_rate)
        upd_R = perspective_taker.update_rotation_matrix_(grad_R, at_learning_rate)

        # zero out gradients for all parameters in all time steps of tuning horizon
        for i in range(tuning_length):
            Bs[i].grad.data.zero_()
            Cs[i].grad.data.zero_()
            Rs[i].grad.data.zero_()
        
        # update all parameters for all time steps 
        for i in range(tuning_length):
            Bs[i] = upd_B
            Cs[i] = upd_C
            Rs[i] = upd_R

        # print(Bs[0])
        # print(Cs[0])
        # print(Rs[0])

        # forward pass from t-H to t with new parameters 
        for i in range(tuning_length):
            o = preprocessor.convert_data_LSTM_to_AT(at_inputs[i])
            x_B = binder.bind(o, Bs[i])
            x_C = perspective_taker.translate(x_B, Cs[i])
            x_R = perspective_taker.rotate(x_C, Rs[i])

            x = preprocessor.convert_data_AT_to_LSTM(x_R)

            # print(f'x_B :{x_B}')
            # print(f'x_C :{x_C}')
            # print(f'x_R :{x_R}')
            # print(f'x :{x}')

            with torch.no_grad():  
                at_predictions[i], state = core_model(x)   
            
    # reorganize storage
    at_inputs = torch.cat((at_inputs[1:], x.reshape(1,45)), 0)
    at_final_predictions = torch.cat((at_final_predictions, at_predictions[0].reshape(1,45)), 0)

    # generate updated prediction 
    with torch.no_grad():
        new_prediction, state = core_model(x)
    at_predictions = torch.cat((at_predictions[1:], new_prediction.reshape(1,45)), 0)
    
# store rest of predictions in at_final_predictions
for i in range(tuning_length): 
    at_final_predictions = torch.cat((at_final_predictions, at_predictions[1].reshape(1,45)), 0)


print(Bs[0])
print(Cs[0])
print(Rs[0])    


#### EVALUATION / VISUALIZATION OF RESULTS 
pred_errors = evaluator.prediction_errors(observations, 
                                          at_final_predictions, 
                                          at_loss_function)

evaluator.help_visualize_devel(observations, at_final_predictions)
