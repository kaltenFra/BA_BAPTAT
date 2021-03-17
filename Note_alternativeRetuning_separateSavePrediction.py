# forward pass from t-H to t with new parameters 
                init_state = (at_h, at_c)
                state = (init_state[0], init_state[1])
                for i in range(self.tuning_length):

                    bm = self.binder.scale_binding_matrix(self.Bs[i], self.scale_mode, self.scale_combo)
                    x_B = self.binder.bind(self.at_inputs[i], bm)
                    x = self.preprocessor.convert_data_AT_to_LSTM(x_B)

                    # print(f'x_B :{x_B}')

                    state = (state[0] * state_scaler, state[1] * state_scaler)
                    
                    # for last tuning cycle update initial state to track gradients 
                    if cycle==(self.tuning_cycles-1):
                        if i==0: 
                            with torch.no_grad():
                                final_prediction = self.at_predictions[0].clone().detach()

                            at_h = state[0].clone().detach().requires_grad_().to(self.device)
                            at_c = state[1].clone().detach().requires_grad_().to(self.device)
                            init_state = (at_h, at_c)
                            state = (init_state[0], init_state[1])
                            
                        else: 
                            self.at_predictions[i-1], state = self.core_model(x, state)
                    
                    else: 
                        self.at_predictions[i], state = self.core_model(x, state)

                    self.at_states[i+1] = state 

                # Update current binding
                bm = self.binder.scale_binding_matrix(self.Bs[-1], self.scale_mode, self.scale_combo)
                x_B = self.binder.bind(o, bm)
                x = self.preprocessor.convert_data_AT_to_LSTM(x_B)


            # END tuning cycle        

            ## Generate updated prediction 
            state = self.at_states[-1]
            state = (state[0] * state_scaler, state[1] * state_scaler)
            self.at_predictions[-1], state = self.core_model(x, state)

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
            # self.at_predictions = torch.cat((self.at_predictions[1:], new_prediction.reshape(1,self.input_per_frame)), 0)