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
from CoreLSTM.core_lstm import PSEUDO_CORE
from Data_Compiler.data_preparation import Preprocessor
from BAPTAT_evaluation import BAPTAT_evaluator

############################################################################
##########  PARAMETERS  ####################################################

## General parameters 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autograd.set_detect_anomaly(True)


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


## Define tuning parameters 
tuning_length = 10      # length of tuning horizon 
tuning_cycles = 1       # number of tuning cycles in each iteration 

# possible loss functions
mse = nn.MSELoss()
l1Loss = nn.L1Loss()
l2Loss = lambda x,y: mse(x, y) * (num_input_dimensions * num_input_features)

# define learning parameters 
at_loss_function = l1Loss
at_learning_rate = 1
at_learning_rate_state = 0.0
c_momentum = 0.0

## Define tuning variables
# general
obs_count = 0
at_inputs = torch.tensor([])
at_predictions = torch.tensor([])
at_final_predictions = torch.tensor([])
at_losses = []

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
observations, feature_names = preprocessor.get_AT_data(data_asf_path, data_amc_path, num_frames+2)
div_observations = preprocessor.get_motion_data(observations, num_frames+2)    


## Load model
core_model = PSEUDO_CORE()


## Translation biases 
tb = perspective_taker.init_translation_bias_()

for i in range(tuning_length+1):
    # transba = copy.deepcopy(tb)
    transba = tb.clone()
    transba.requires_grad = True
    Cs.append(transba)

print(f'BMs different in list: {Cs[0] is not Cs[1]}')


############################################################################
##########  FORWARD PASS  ##################################################

for i in range(tuning_length):
    o = observations[obs_count]
    div_o_next = div_observations[obs_count]
    at_inputs = torch.cat((at_inputs, o.reshape(1, num_input_features, num_input_dimensions)), 0)
    obs_count += 1

    x_C = perspective_taker.translate(o, Cs[i])

    new_prediction = core_model.forward(x_C, Cs[i].clone().detach(), o.clone().detach(), div_o_next)  
    at_predictions = torch.cat((at_predictions, new_prediction.reshape(1,num_input_features, num_input_dimensions)), 0)

############################################################################
##########  ACTIVE TUNING ##################################################

while obs_count < num_frames:
    # TODO folgendes evtl in function auslagern
    o = observations[obs_count]
    div_o_next = div_observations[obs_count]
    obs_count += 1

    # x_C = perspective_taker.translate(o, Cs[-1]/c_norm)
    x_C = perspective_taker.translate(o, Cs[-1])

    ## Generate current prediction 
    with torch.no_grad():
        new_prediction = core_model.forward(x_C, Cs[i].clone().detach(), o.clone().detach(), div_o_next)  

    ## For #tuning_cycles 
    for cycle in range(tuning_cycles):
        print('----------------------------------------------')

        # Get prediction
        p = at_predictions[-1]

        # Calculate error 
        loss_scale_factor = 0.5
        loss = loss_scale_factor * l2Loss(p,x_C[0]) + mse(p,x_C[0])
        # loss = loss_scale_factor * l1Loss(p,x_C[0]) + mse(p,x_C[0])

        at_losses.append(loss)
        print(f'frame: {obs_count} cycle: {cycle} loss: {loss}')

        # Propagate error back through tuning horizon 
        loss.backward(retain_graph = True)

        # Update parameters 
        with torch.no_grad():
            ## Normalization factor
            c_norm = loss.detach()
            
            ## Binding Matrix
            for i in range(tuning_length+1):
                # save grads for all parameters in all time steps of tuning horizon 
                C_grads[i] = Cs[i].grad
                # print(Cs[i].grad)
            # print(C_grads[tuning_length])
            
            # Calculate overall gradients 
            ### version 1
            # grad_C = C_grads[0]
            ### version 2 / 3
            grad_C = torch.mean(torch.stack(C_grads))
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

        
        ## REORGANIZE FOR MULTIPLE CYCLES!!!!!!!!!!!!!

        # forward pass from t-H to t with new parameters 
        for i in range(tuning_length):

            # x_C = perspective_taker.translate(at_inputs[i], Cs[i]/c_norm)
            x_C = perspective_taker.translate(at_inputs[i], Cs[i])

            # print(f'x_B :{x_B}')
            div_o = div_observations[obs_count-1-tuning_length+i]
            at_predictions[i] = core_model.forward(x_C, Cs[i].clone().detach(), o.clone().detach(), div_o)  

        # Update current binding
        # x_C = perspective_taker.translate(o, Cs[-1]/c_norm) 
        x_C = perspective_taker.translate(o, Cs[-1]) 

    # END tuning cycle        

    ## Generate updated prediction 
    new_prediction = core_model.forward(x_C, Cs[-1].clone().detach(), o.clone().detach(), div_o_next)

    ## Reorganize storage variables
   
    # observations
    at_inputs = torch.cat((at_inputs[1:], o.reshape(1, num_input_features, num_input_dimensions)), 0)
    
    # predictions
    at_final_predictions = torch.cat((at_final_predictions, at_predictions[0].detach().reshape(1,45)), 0)
    at_predictions = torch.cat((at_predictions[1:],  new_prediction.reshape(1,num_input_features, num_input_dimensions)), 0)

# END active tuning
    
# store rest of predictions in at_final_predictions
for i in range(tuning_length): 
    at_final_predictions = torch.cat((at_final_predictions, at_predictions[1].reshape(1,45)), 0)

# get final binding matrix
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
