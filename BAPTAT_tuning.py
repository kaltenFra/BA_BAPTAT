# packet imports 
import torch
from torch import autograd

## General parameters 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autograd.set_detect_anomaly(True)

class BAPTAT_tuning():
    def __init__(self, at_length, at_cycles, at_loss_function, at_learning_rate, 
                 num_frames, num_features, num_dimensions, observations, model, 
                 preprocessor, binder, sigmoidal, perspective_taker):

        self.tuning_length = at_length
        self.tuning_cycles = at_cycles
        self.loss_function = at_loss_function
        self.learning_rate = at_learning_rate
        self.num_frames = num_frames
        self.num_input_features = num_features
        self.num_input_dimensions = num_dimensions
        self.observations = observations
        self.core_model = model
        self.preprocessor = preprocessor
        self.binder = binder
        self.bindSM = sigmoidal
        self.perspective_taker = perspective_taker


    def observation_to_input(self, obs, binding_matrix, translation_bias, rotation_matrix):
        x_B = self.binder.bind(obs, self.bindSM(binding_matrix))
        x_C = self.perspective_taker.translate(x_B, translation_bias)
        x_R = self.perspective_taker.rotate(x_C, rotation_matrix)
        x = self.preprocessor.convert_data_AT_to_LSTM(x_R)
        return x


    def tune(self):
        ## Create list of parameters for gradient descent
        Bs = []
        Cs = []
        Rs = []
        B_grads = [None] * self.tuning_length
        C_grads = [None] * self.tuning_length
        R_grads = [None] * self.tuning_length

        for i in range(self.tuning_length):
            Bs.append(self.binder.binding_matrix_())
            Cs.append(self.perspective_taker.translation_bias_())
            Rs.append(self.perspective_taker.rotation_matrix_())


        #### FORWARD PASS FOR ONE TUNING HORIZON --- NOT closed loop! preds based on observations
        obs_count = 0
        at_inputs = torch.tensor([])
        at_predictions = torch.tensor([])
        at_final_predictions = torch.tensor([])
        at_losses = []

        for i in range(self.tuning_length):
            o = self.observations[obs_count]
            obs_count += 1

            x = self.observation_to_input(o, Bs[i], Cs[i], Rs[i])

            at_inputs = torch.cat((at_inputs, x), 0)
            
            with torch.no_grad():
                new_prediction, state = self.core_model(x)
                at_predictions = torch.cat((at_predictions, new_prediction.reshape(1,45)), 0)


        #### ACTIVE TUNING PART 

        while obs_count < self.num_frames:
            o = self.observations[obs_count]
            obs_count += 1
            x = self.observation_to_input(o, Bs[-1], Cs[-1], Rs[-1])

            ## Generate current prediction 
            with torch.no_grad():
                new_prediction, state = self.core_model(x)

            ## For #tuning_cycles 
            for cycle in range(self.tuning_cycles):
                print('----------------------------------------------')

                # get prediction
                p = at_predictions[-1]

                # calculate error 
                loss = self.at_loss_function(p, x)
                at_losses.append(loss)
                print(f'frame: {obs_count} cycle: {cycle} loss: {loss}')

                # propagate error back through tuning horizon 
                loss.backward(retain_graph=True)

                # save grads for all parameters in all time steps of tuning horizon 
                for i in range(self.tuning_length):
                    B_grads[i] = Bs[i].grad
                    C_grads[i] = Cs[i].grad
                    R_grads[i] = Rs[i].grad
                
                # calculate overall gradients 
                grad_B = torch.mean(torch.stack(B_grads))
                grad_C = torch.mean(torch.stack(C_grads))
                grad_R = torch.mean(torch.stack(R_grads))

                # update parameters in time step t-H with saved gradients 
                upd_B = self.binder.update_binding_matrix_(grad_B, self.at_learning_rate)
                upd_C = self.perspective_taker.update_translation_bias_(grad_C, self.at_learning_rate)
                upd_R = self.perspective_taker.update_rotation_matrix_(grad_R, self.at_learning_rate)

                # zero out gradients for all parameters in all time steps of tuning horizon
                for i in range(self.tuning_length):
                    Bs[i].grad.data.zero_()
                    Cs[i].grad.data.zero_()
                    Rs[i].grad.data.zero_()
                
                # update all parameters for all time steps 
                for i in range(self.tuning_length):
                    Bs[i] = upd_B
                    Cs[i] = upd_C
                    Rs[i] = upd_R

                # forward pass from t-H to t with new parameters 
                for i in range(self.tuning_length):
                    o = self.preprocessor.convert_data_LSTM_to_AT(at_inputs[i])
                    x = self.observation_to_input(o, Bs[i], Cs[i], Rs[i])

                    with torch.no_grad():  
                        at_predictions[i], state = self.core_model(x)   
                    
            # reorganize storage
            at_inputs = torch.cat((at_inputs[1:], x.reshape(1,45)), 0)
            at_final_predictions = torch.cat((at_final_predictions, at_predictions[0].reshape(1,45)), 0)

            # generate updated prediction 
            with torch.no_grad():
                new_prediction, state = self.core_model(x)
            at_predictions = torch.cat((at_predictions[1:], new_prediction.reshape(1,45)), 0)
            
        # store rest of predictions in at_final_predictions
        for i in range(self.tuning_length): 
            at_final_predictions = torch.cat((at_final_predictions, at_predictions[1].reshape(1,45)), 0)

        final_binding_matrix = Bs[0]
        final_translation_bias = Cs[0]
        final_rotation_matrix = Rs[0]

        return final_binding_matrix, final_translation_bias, final_rotation_matrix, at_final_predictions, at_losses

