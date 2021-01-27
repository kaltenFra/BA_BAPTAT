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
tuning_cycles = 1       # number of tuning cycles in each iteration 
at_loss_function = nn.MSELoss()

at_final_predictions = torch.tensor([])
B_grads = [None] * tuning_length

at_learning_rate = 0.0001


## Define data parameters
data_at_unlike_train = True
num_frames = 800
num_input_features = 15
num_input_dimensions = 3
preprocessor = Preprocessor(num_input_features, num_input_dimensions)
evaluator = BAPTAT_evaluator(num_frames, preprocessor)

# data paths 
data_asf_path = 'Data_Compiler/S35T07.asf'
data_amc_path = 'Data_Compiler/S35T07.amc'

####################### Note: sample needs to be changed in the future

## Define model parameters 
model_path = 'CoreLSTM/models/LSTM_6.pt'

## Define Binding parameters 
# Binding
bindSM = nn.Softmax(dim=0)  # columnwise
# bindSM = nn.Softmax(dim=1)  # rowwise
binder = Binder(num_features=num_input_features, gradient_init=True)


## Create list of parameters for gradient descent
Bs = []
################################################### appending separate matrices or only references?
###################### maybe tensors not lists? -> backprop possible 
for i in range(tuning_length):
    Bs.append(binder.binding_matrix_())

# print(Bs[0])

## Load data
observations, feature_names = preprocessor.get_AT_data(data_asf_path, data_amc_path, num_frames)
    
## Load model
core_model = CORE_NET()
core_model.load_state_dict(torch.load(model_path))
core_model.eval()

#### FORWARD PASS FOR ONE TUNING HORIZON --- NOT closed loop! preds based on observations
obs_count = 0
at_inputs = torch.tensor([])
at_predictions = torch.tensor([])
at_losses = []

for i in range(tuning_length):
    o = observations[obs_count]
    obs_count += 1

    x_B = binder.bind(o, bindSM(Bs[i]))

    x = preprocessor.convert_data_AT_to_LSTM(x_B)

    at_inputs = torch.cat((at_inputs, x), 0)
    
    with torch.no_grad():
        new_prediction, state = core_model(x)
        at_predictions = torch.cat((at_predictions, new_prediction.reshape(1,45)), 0)


#### ACTIVE TUNING PART 

while obs_count < num_frames:
    # TODO folgendes evtl in function auslagern
    o = observations[obs_count]
    obs_count += 1

    x_B = binder.bind(o, bindSM(Bs[-1]))

    x = preprocessor.convert_data_AT_to_LSTM(x_B)

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
        at_losses.append(loss)
        print(f'frame: {obs_count} cycle: {cycle} loss: {loss}')

        # propagate error back through tuning horizon 
        loss.backward(retain_graph=True)
        ############################# retain_graph = True ? 
        ############################# optimizer? Adam.. 

        # for i in TH
        for i in range(tuning_length):
            # save grads for all parameters in all time steps of tuning horizon 
            B_grads[i] = Bs[i].grad
            # Working? not all tensors.. maybe add Varables? --> added Variables  
        
        # calculate overall gradients 
        grad_B = torch.mean(torch.stack(B_grads))

        # update parameters in time step t-H with saved gradients 
        upd_B = binder.update_binding_matrix_(grad_B, at_learning_rate)

        # zero out gradients for all parameters in all time steps of tuning horizon
        for i in range(tuning_length):
            Bs[i].grad.data.zero_()
        
        # update all parameters for all time steps 
        for i in range(tuning_length):
            Bs[i] = upd_B

        # print(Bs[0])

        # forward pass from t-H to t with new parameters 
        for i in range(tuning_length):
            o = preprocessor.convert_data_LSTM_to_AT(at_inputs[i])
            x_B = binder.bind(o, bindSM(Bs[i]))

            x = preprocessor.convert_data_AT_to_LSTM(x_B)

            # print(f'x_B :{x_B}')

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

final_binding_matrix = Bs[0]

print(f'final binding matrix: {final_binding_matrix}')


#### EVALUATION / VISUALIZATION OF RESULTS 
pred_errors = evaluator.prediction_errors(observations, 
                                          at_final_predictions, 
                                          at_loss_function)

evaluator.plot_at_losses(at_losses)

evaluator.plot_binding_matrix(final_binding_matrix, feature_names)

evaluator.help_visualize_devel(observations, at_final_predictions)
