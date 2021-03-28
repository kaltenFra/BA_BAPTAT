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



class SEP_ROTATION():

    def __init__(self):
        ## General parameters 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autograd.set_detect_anomaly(True)

        torch.set_printoptions(precision=8)

        ## Set default parameters
        ##   -> Can be changed during experiments 

        self.rotation_type = 'qrotate'
        self.grad_bias = 1.5

    
    ############################################################################
    ##########  PARAMETERS  ####################################################

    def set_rotation_type(self, rotation): 
        self.rotation_type = rotation
        print('Reset type of rotation: ' + self.rotation_type)


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

        self.perspective_taker = Perspective_Taker(
            self.num_observations,
            self.num_input_dimensions,
            rotation_gradient_init=True, 
            translation_gradient_init=True)

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
        at_learning_rate_rotation, 
        at_learning_rate_state, 
        at_momentum_rotation): 

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
        self.at_learning_rate = at_learning_rate_rotation
        self.at_learning_rate_state = at_learning_rate_state
        self.r_momentum = at_momentum_rotation
        self.at_loss_function = self.mse

        print('Parameters set.')


    def init_model_(self, model_path): 
        ## Load model
        self.core_model = CORE_NET(45, 150)
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
        self.at_final_inputs = torch.tensor([]).to(self.device)
        self.at_final_predictions = torch.tensor([]).to(self.device)
        self.at_losses = []

        # state 
        self.at_states = []
        # state_optimizer = torch.optim.Adam(init_state, at_learning_rate)

        # rotation
        self.Rs = []
        self.R_grads = [None] * (self.tuning_length+1)
        self.R_upd = [None] * (self.tuning_length+1)
        self.rm_losses = []
        self.rv_losses = []
        self.ra_losses = []

    
    def set_comparison_values(self, ideal_rotation_values, ideal_rotation_matrix):
        self.identity_matrix = torch.Tensor(np.identity(self.num_input_dimensions))
        self.ideal_rotation = ideal_rotation_matrix.to(self.device)
        if self.rotation_type == 'qrotate': 
            self.ideal_quat = ideal_rotation_values.to(self.device)
            self.ideal_angle = torch.rad2deg(self.perspective_taker.qeuler(self.ideal_quat, 'xyz')).to(self.device)
        elif self.rotation_type == 'eulrotate': 
            self.ideal_angle = ideal_rotation_values.to(self.device)
        else: 
            print(f'ERROR: Received unknown rotation type!\n\trotation type: {self.rotation_type}')
            exit()


    def set_ideal_euler_angles(self, angles): 
        self.ideal_angles = angles.to(self.device)     


    def set_ideal_quaternion(self, quaternion): 
        self.ideal_quat = quaternion.to(self.device)

    
    def set_ideal_roation_matrix(self, matrix):
        self.ideal_rotation = matrix.to(self.device)


    ############################################################################
    ##########  INFERENCE  #####################################################
    
    def run_inference(self, observations, grad_calculation):

        at_final_predictions = torch.tensor([]).to(self.device)
        at_final_inputs = torch.tensor([]).to(self.device)

        if self.rotation_type == 'qrotate': 
            ## Rotation quaternion 
            rq = self.perspective_taker.init_quaternion()
            # print(rq)

            for i in range(self.tuning_length+1):
                quat = rq.clone().to(self.device)
                quat.requires_grad_()
                self.Rs.append(quat)

        elif self.rotation_type == 'eulrotate':
            ## Rotation euler angles 
            # ra = perspective_taker.init_angles_()
            # ra = torch.Tensor([[309.89], [82.234], [95.765]])
            ra = torch.Tensor([[75.0], [6.0], [128.0]])
            print(ra)

            for i in range(self.tuning_length+1):
                angles = []
                for j in range(self.num_input_dimensions):
                    angle = ra[j].clone()
                    angle.requires_grad_()
                    angles.append(angle)
                self.Rs.append(angles)

        else: 
            print('ERROR: Received unknown rotation type!')
            exit()


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
            self.at_inputs = torch.cat((self.at_inputs, o.reshape(1, self.num_input_features, self.num_input_dimensions)), 0)
            self.obs_count += 1

            if self.rotation_type == 'qrotate': 
                x_R = self.perspective_taker.qrotate(o, self.Rs[i])
            else:
                rotmat = self.perspective_taker.compute_rotation_matrix_(self.Rs[i][0], self.Rs[i][1], self.Rs[i][2])
                x_R = self.perspective_taker.rotate(o, rotmat)
            
            x = self.preprocessor.convert_data_AT_to_LSTM(x_R)

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

            if self.rotation_type == 'qrotate': 
                x_R = self.perspective_taker.qrotate(o, self.Rs[-1])

            else:
                rotmat = self.perspective_taker.compute_rotation_matrix_(self.Rs[-1][0], self.Rs[-1][1], self.Rs[-1][2])
                x_R = self.perspective_taker.rotate(o, rotmat)

            x = self.preprocessor.convert_data_AT_to_LSTM(x_R)

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

                # Propagate error back through tuning horizon 
                loss.backward(retain_graph = True)

                self.at_losses.append(loss.clone().detach().cpu().numpy())
                print(f'frame: {self.obs_count} cycle: {cycle} loss: {loss}')

                # Update parameters 
                with torch.no_grad():
                    
                    ## get gradients
                    if self.rotation_type == 'qrotate': 
                        for i in range(self.tuning_length+1):
                            # save grads for all parameters in all time steps of tuning horizon
                            self.R_grads[i] = self.Rs[i].grad
                    else: 
                        for i in range(self.tuning_length+1):
                            # save grads for all parameters in all time steps of tuning horizon
                            grad = []
                            for j in range(self.num_input_dimensions):
                                grad.append(self.Rs[i][j].grad) 
                            self.R_grads[i] = torch.stack(grad)

                    # print(self.R_grads[self.tuning_length])
                    
                    # Calculate overall gradients 
                    if grad_calculation == 'lastOfTunHor':
                        ### version 1
                        grad_R = self.R_grads[0]
                    elif grad_calculation == 'meanOfTunHor':
                        ### version 2 / 3
                        grad_R = torch.mean(torch.stack(self.R_grads), dim=0)
                    elif grad_calculation == 'weightedInTunHor':
                        ### version 4
                        weighted_grads_R = [None] * (self.tuning_length+1)
                        for i in range(self.tuning_length+1):
                            weighted_grads_R[i] = np.power(self.grad_bias, i) * self.R_grads[i]
                        grad_R = torch.mean(torch.stack(weighted_grads_R), dim=0)
                    
                    # print(f'grad_R: {grad_R}')

                    grad_R = grad_R.to(self.device)
                    if self.rotation_type == 'qrotate': 
                        # Update parameters in time step t-H with saved gradients 
                        upd_R = self.perspective_taker.update_quaternion(
                            self.Rs[0], grad_R, self.at_learning_rate, self.r_momentum)
                        print(f'updated quaternion: {upd_R}')

                        # Compare quaternion values
                        # quat_loss = torch.sum(self.perspective_taker.qmul(self.ideal_quat, upd_R))
                        quat_loss = 2 * torch.arccos(torch.abs(torch.sum(torch.mul(self.ideal_quat, upd_R))))
                        quat_loss = torch.rad2deg(quat_loss)
                        print(f'loss of quaternion: {quat_loss}')
                        self.rv_losses.append(quat_loss)

                        # Compare quaternion angles
                        ang = torch.rad2deg(self.perspective_taker.qeuler(upd_R, 'zyx'))
                        ang_diff = ang - self.ideal_angle
                        ang_loss = 2 - (torch.cos(torch.deg2rad(ang_diff)) + 1)
                        print(f'loss of quaternion angles: {ang_loss} \nwith norm: {torch.norm(ang_loss)}')
                        self.ra_losses.append(torch.norm(ang_loss))
                        
                        # Compute rotation matrix
                        rotmat = self.perspective_taker.quaternion2rotmat(upd_R)

                        # Zero out gradients for all parameters in all time steps of tuning horizon
                        for i in range(self.tuning_length+1):
                            self.Rs[i].requires_grad = False
                            self.Rs[i].grad.data.zero_()

                        # Update all parameters for all time steps 
                        for i in range(self.tuning_length+1):
                            quat = upd_R.clone()
                            quat.requires_grad_()
                            self.Rs[i] = quat

                    else: 
                        # Update parameters in time step t-H with saved gradients 
                        upd_R = self.perspective_taker.update_rotation_angles_(self.Rs[0], grad_R, self.at_learning_rate)
                        print(f'updated angles: {upd_R}')

                        # Save rotation angles
                        rotang = torch.stack(upd_R)
                        # angles:
                        ang_diff = rotang - self.ideal_angle
                        ang_loss = 2 - (torch.cos(torch.deg2rad(ang_diff)) + 1)
                        print(f'loss of rotation angles: \n  {ang_loss}, \n  with norm {torch.norm(ang_loss)}')
                        self.rv_losses.append(torch.norm(ang_loss))
                        # Compute rotation matrix
                        rotmat = self.perspective_taker.compute_rotation_matrix_(upd_R[0], upd_R[1], upd_R[2])[0]
                        
                        # Zero out gradients for all parameters in all time steps of tuning horizon
                        for i in range(self.tuning_length+1):
                            for j in range(self.num_input_dimensions):
                                self.Rs[i][j].requires_grad = False
                                self.Rs[i][j].grad.data.zero_()

                        # print(Rs[0])
                        # Update all parameters for all time steps 
                        for i in range(self.tuning_length+1):
                            angles = []
                            for j in range(3):
                                angle = upd_R[j].clone()
                                angle.requires_grad_()
                                angles.append(angle)
                            self.Rs[i] = angles
                        # print(Rs[0])

                    # Calculate and save rotation losses
                    # matrix: 
                    # mat_loss = self.mse(
                    #     (torch.mm(self.ideal_rotation, torch.transpose(rotmat, 0, 1))), 
                    #     self.identity_matrix
                    # )
                    dif_R = torch.mm(self.ideal_rotation, torch.transpose(rotmat, 0, 1))
                    mat_loss = torch.arccos(0.5 * (torch.trace(dif_R)-1))
                    mat_loss = torch.rad2deg(mat_loss)

                    print(f'loss of rotation matrix: {mat_loss}')
                    self.rm_losses.append(mat_loss)
                    

                    # print(Rs[0])

                    # Initial state
                    g_h = at_h.grad.to(self.device)
                    g_c = at_c.grad.to(self.device)

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
                self.at_predictions = torch.tensor([]).to(self.device)
                for i in range(self.tuning_length):

                    if self.rotation_type == 'qrotate': 
                        x_R = self.perspective_taker.qrotate(self.at_inputs[i], self.Rs[i])
                    else:
                        rotmat = self.perspective_taker.compute_rotation_matrix_(self.Rs[i][0], self.Rs[i][1], self.Rs[i][2])
                        x_R = self.perspective_taker.rotate(self.at_inputs[i], rotmat)

                    x = self.preprocessor.convert_data_AT_to_LSTM(x_R)

                    state = (state[0] * state_scaler, state[1] * state_scaler)
                    upd_prediction, state = self.core_model(x, state)
                    self.at_predictions = torch.cat((self.at_predictions, upd_prediction.reshape(1,self.input_per_frame)), 0)
                    
                    # for last tuning cycle update initial state to track gradients 
                    if cycle==(self.tuning_cycles-1) and i==0: 
                        with torch.no_grad():
                            final_prediction = self.at_predictions[0].clone().detach().to(self.device)
                            final_input = x.clone().detach().to(self.device)

                        at_h = state[0].clone().detach().requires_grad_()
                        at_c = state[1].clone().detach().requires_grad_()
                        init_state = (at_h, at_c)
                        state = (init_state[0], init_state[1])

                    self.at_states[i] = state 

                # Update current rotation
                if self.rotation_type == 'qrotate': 
                    x_R = self.perspective_taker.qrotate(o, self.Rs[-1])
                else:
                    rotmat = self.perspective_taker.compute_rotation_matrix_(self.Rs[-1][0], self.Rs[-1][1], self.Rs[-1][2])
                    x_R = self.perspective_taker.rotate(o, rotmat)

                x = self.preprocessor.convert_data_AT_to_LSTM(x_R)

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
                o.reshape(1, self.num_input_features, self.num_input_dimensions)), 0)
            
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
            if self.rotation_type == 'qrotate': 
                x_i = self.perspective_taker.qrotate(self.at_inputs[i], self.Rs[-1])
            else: 
                x_i = self.perspective_taker.rotate(self.at_inputs[i], rotmat)
            at_final_inputs = torch.cat(
                (at_final_inputs, 
                x_i.reshape(1,self.input_per_frame)), 0)


        # get final rotation matrix
        if self.rotation_type == 'qrotate': 
            final_rotation_values = self.Rs[0].clone().detach()
            # get final quaternion
            print(f'final quaternion: {final_rotation_values}')
            final_rotation_matrix = self.perspective_taker.quaternion2rotmat(final_rotation_values)
            
        else:
            final_rotation_values = [self.Rs[0][i].clone().detach() for i in range(self.num_input_dimensions)]
            print(f'final euler angles: {final_rotation_values}')
            final_rotation_matrix = self.perspective_taker.compute_rotation_matrix_(
                final_rotation_values[0], 
                final_rotation_values[1], 
                final_rotation_values[2])
        
        print(f'final rotation matrix: \n{final_rotation_matrix}')

        return at_final_inputs, at_final_predictions, final_rotation_values, final_rotation_matrix


    ############################################################################
    ##########  EVALUATION #####################################################

    def get_result_history(
        self, 
        observations, 
        at_final_predictions):

        pred_errors = self.evaluator.prediction_errors(observations, 
                                                at_final_predictions, 
                                                self.mse)
        print([pred_errors, self.at_losses, self.rm_losses, self.rv_losses, self.ra_losses])

        return [pred_errors, self.at_losses, self.rm_losses, self.rv_losses, self.ra_losses]



