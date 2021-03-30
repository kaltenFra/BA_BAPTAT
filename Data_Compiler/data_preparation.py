from numpy.core.numeric import Inf
from torch.functional import Tensor, norm
import torch 
import numpy as np
import math

import sys
sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from Data_Compiler.amc_parser import test_all

class Preprocessor():
    
#     def __init__(self, num_features, num_dimensions):
#         self._num_features = num_features
#         self.num_observations = num_features
#         self._num_dimensions = num_dimensions


#     def compile_data(self, asf_path, amc_path, frame_samples):
#         visual_input, selected_joint_names = test_all(asf_path, amc_path, frame_samples, 30, self.num_observations) 
#         visual_input = torch.from_numpy(visual_input).type(torch.float)
        
#         return visual_input, selected_joint_names
    

#     def std_scale_data(self, input_data, factor):
#         normed = torch.norm(input_data, dim=2)
#         scale_factor = 1/(np.sqrt(factor) * normed.std())
#         scale_mat = torch.Tensor([[scale_factor, 0, 0], 
#                                   [0, scale_factor, 0], 
#                                   [0, 0, scale_factor]])
#         scaled = torch.matmul(input_data, scale_mat)
#         print(f'Scaled data by factor {scale_factor}')
#         print(f'New minimum: {torch.min(scaled)}')
#         print(f'New maximum: {torch.max(scaled)}')
#         return scaled


#     def create_inout_sequences(self, input_data, tw):
#         inout_seq = []
#         L = len(input_data)
#         for i in range(L-tw):
#             train_seq = input_data[i:i+tw]
#             train_label = input_data[i+tw:i+tw+1]
#             inout_seq.append((train_seq ,train_label))
#         return inout_seq




#     def get_LSTM_data(self, asf_path, amc_path, frame_samples, num_test_data, train_window):
#         visual_input, selected_joint_names = self.compile_data(asf_path=asf_path, amc_path=amc_path, frame_samples=frame_samples)

#         visual_input = visual_input.permute(1,0,2)
#         visual_input = self.std_scale_data(visual_input, 15)
#         visual_input = visual_input.reshape(1, frame_samples, self._num_dimensions*self._num_features)

#         train_data = visual_input[:,:-num_test_data,:]
#         test_data = visual_input[:,-num_test_data:,:]

#         train_inout_seq = self.create_inout_sequences(train_data[0], train_window)

#         # train_data = train_data.reshape(frame_samples-num_test_data, self._num_features, self._num_dimensions)

#         return train_inout_seq, train_data, test_data

    


#     def get_LSTM_data_motion(self, asf_path, amc_path, frame_samples, num_test_data, train_window):
#         visual_input, selected_joint_names = self.compile_data(asf_path=asf_path, amc_path=amc_path, frame_samples=frame_samples)

#         visual_input = visual_input.permute(1,0,2)
#         visual_input = self.std_scale_data(visual_input, 1)
#         visual_input = self.get_motion_data(visual_input, frame_samples)
#         visual_input = visual_input.reshape(1, frame_samples-1, self._num_dimensions*self._num_features)

#         train_data = visual_input[:,:-num_test_data,:]
#         test_data = visual_input[:,-num_test_data:,:]

#         train_inout_seq = self.create_inout_sequences(train_data[0], train_window)

#         # train_data = train_data.reshape(frame_samples-num_test_data, self._num_features, self._num_dimensions)

#         return train_inout_seq, train_data, test_data

    
#     def get_AT_data(self, asf_path, amc_path, frame_samples):
#         visual_input, selected_joint_names = self.compile_data(asf_path=asf_path, amc_path=amc_path, frame_samples=frame_samples)
#         visual_input = visual_input.permute(1,0,2)
#         visual_input = self.std_scale_data(visual_input, 15)


#         return visual_input, selected_joint_names
    

#     def convert_data_AT_to_LSTM(self, data):
#         return data.reshape(1, self._num_dimensions*self._num_features)
#         # return data.reshape(1, num_samples, self._num_dimensions*self._num_features)


#     def convert_data_LSTM_to_AT(self, data): 
#         return data.reshape(self._num_features, self._num_dimensions)
#         # return data.reshape(num_samples, self._num_features, self._num_dimensions)

# class Preprocessor_nxm():
    
    def __init__(self, num_observations=None, num_features=15, num_dimensions=3, distractor=False):
        self._num_features = num_features
        if num_observations is None: 
            self.num_observations = num_features
            self.distractor = False
        else:
            self.num_observations = num_observations
            self.distractor = distractor
            
        self._num_dimensions = num_dimensions
        self.num_spatial_dimensions = 3


    def reset_dimensions(self, dim):
        self._num_dimensions = dim
        print(f'Reset dimensions to {self._num_dimensions}')

        
    def compile_data(self, asf_path, amc_path, frame_samples):
        visual_input, selected_joint_names = test_all(asf_path, amc_path, frame_samples, 30, self.num_observations) 
        visual_input = torch.from_numpy(visual_input).type(torch.float)
        
        return visual_input, selected_joint_names
    

    def std_scale_data(self, input_data, factor):
        normed = torch.norm(input_data, dim=2)
        scale_factor = 1/(np.sqrt(factor) * normed.std())
        scale_mat = torch.Tensor([[scale_factor, 0, 0], 
                                  [0, scale_factor, 0], 
                                  [0, 0, scale_factor]])
        scaled = torch.matmul(input_data, scale_mat)
        print(f'Scaled data by factor {scale_factor}')
        print(f'New minimum: {torch.min(scaled)}')
        print(f'New maximum: {torch.max(scaled)}')
        return scaled


    def create_inout_sequences(self, input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append((train_seq ,train_label))
        return inout_seq


    def get_motion_data(self, abs_data, num_frames):
        motion_dt = torch.Tensor(num_frames-1, self.num_observations, self.num_spatial_dimensions)
        for i in range(num_frames-1):
            # motion_dt[i] = abs_data[i+1] - abs_data[i]
            motion_dt[i] = abs_data[i] - abs_data[i+1]
            # NOTE: try with causal time (t-1) - (t)
        # print('Constructed motion data.')
        return motion_dt

    
    def get_direction_data(self, abs_data, num_frames): 
        velocity = self.get_motion_data(abs_data, num_frames)
        magnitude = self.get_magnitude_data(abs_data, num_frames)
        # direction = torch.Tensor([])

        direction = torch.nn.functional.normalize(velocity, dim=2)

        # for i in range(num_frames-1):
        #     # ds = torch.div(velocity[i].t(),magnitude[i]).t()
        #     ds = torch.nn.normalize(velocity[i])
        #     # ds *= 0.1

        #     # check length of direction vector
        #     # print(torch.sqrt(torch.sum(torch.mul(ds, ds), dim=1)))
        #     # print(ds)
        #     # exit()

        #     direction = torch.cat([direction, ds.view(1,self._num_features,self._num_dimensions)])

        return direction


    def get_magnitude_data(self, abs_data, num_frames): 
        velocity = self.get_motion_data(abs_data, num_frames)
        magnitude = torch.norm(velocity, dim=2)                   # dir vectors with len =1
        # magnitude = torch.linalg.norm(velocity, dim=2, ord=0)     # dir vectors with len <1
        # magnitude = torch.linalg.norm(velocity, dim=2, ord=1)     # dir vectors with len <1
        # magnitude = torch.linalg.norm(velocity, dim=2, ord=Inf)     # dir vectors with len <1
        # magnitude = torch.linalg.norm(velocity, dim=2, ord=-1)    # dir vectors with len >>1
        # magnitude = torch.linalg.norm(velocity, dim=2, ord=-2)    # dir vectors with len >1
        return magnitude

    # def get_motion_data(self, abs_data, num_frames):
    #     motion_dt = torch.Tensor(num_frames-1, self._num_features, self._num_dimensions)
    #     for i in range(num_frames-1):
    #         motion_dt[i] = abs_data[i+1] - abs_data[i]
    #     print('Constructed motion data.')
    #     return motion_dt


    def get_LSTM_data(self, asf_path, amc_path, frame_samples, num_test_data, train_window):
        visual_input, selected_joint_names = self.compile_data(asf_path=asf_path, amc_path=amc_path, frame_samples=frame_samples)

        visual_input = visual_input.permute(1,0,2)
        visual_input = self.std_scale_data(visual_input, 15)

        print(visual_input.shape)
        visual_input = visual_input.reshape(1, frame_samples, self._num_dimensions*self._num_features)

        train_data = visual_input[:,:-num_test_data,:]
        test_data = visual_input[:,-num_test_data:,:]

        train_inout_seq = self.create_inout_sequences(train_data[0], train_window)

        # train_data = train_data.reshape(frame_samples-num_test_data, self._num_features, self._num_dimensions)

        return train_inout_seq, train_data, test_data

    
    def get_gestalt_dir_mag(self, input, frame_samples):
        direction = self.get_direction_data(input, frame_samples)
        magnitude = self.get_magnitude_data(input, frame_samples)
        magnitude = magnitude.view(frame_samples-1, self.num_observations, 1)

        return torch.cat([input[1:], direction, magnitude], dim=2)


    def get_gestalt_vel(self, input, frame_samples):
        velocity = self.get_motion_data(input, frame_samples)

        return torch.cat([input[1:], velocity], dim=2)



    def get_LSTM_data_gestalten(self, asf_path, amc_path, frame_samples, num_test_data, train_window):
        visual_input, selected_joint_names = self.compile_data(asf_path=asf_path, amc_path=amc_path, frame_samples=frame_samples)
        visual_input = visual_input.permute(1,0,2)
        visual_input = self.std_scale_data(visual_input, 15)

        if self._num_dimensions == 6:
            visual_input = self.get_gestalt_vel(visual_input, frame_samples)
        elif self._num_dimensions == 7:
            visual_input = self.get_gestalt_dir_mag(visual_input, frame_samples)

        visual_input = visual_input.reshape(1, frame_samples-1, (self._num_dimensions) *self._num_features)

        train_data = visual_input[:,:-num_test_data,:]
        test_data = visual_input[:,-num_test_data:,:]

        train_inout_seq = self.create_inout_sequences(train_data[0], train_window)

        # train_data = train_data.reshape(frame_samples-num_test_data, self._num_features, self._num_dimensions)

        return train_inout_seq, train_data, test_data


    # def get_LSTM_data_motion(self, asf_path, amc_path, frame_samples, num_test_data, train_window):
    #     visual_input, selected_joint_names = self.compile_data(asf_path=asf_path, amc_path=amc_path, frame_samples=frame_samples)

    #     visual_input = visual_input.permute(1,0,2)
    #     visual_input = self.std_scale_data(visual_input, 1)
    #     visual_input = self.get_motion_data(visual_input, frame_samples)
    #     visual_input = visual_input.reshape(1, frame_samples-1, self._num_dimensions*self._num_features)

    #     train_data = visual_input[:,:-num_test_data,:]
    #     test_data = visual_input[:,-num_test_data:,:]

    #     train_inout_seq = self.create_inout_sequences(train_data[0], train_window)

    #     # train_data = train_data.reshape(frame_samples-num_test_data, self._num_features, self._num_dimensions)

    #     return train_inout_seq, train_data, test_data

    
    def get_AT_data(self, asf_path, amc_path, frame_samples):
        visual_input, selected_joint_names = self.compile_data(asf_path=asf_path, amc_path=amc_path, frame_samples=frame_samples)
        visual_input = visual_input.permute(1,0,2)
        visual_input = self.std_scale_data(visual_input, 15)

        if self.distractor:
            point_position = self.get_distractor_position(frame_samples)
            visual_input = torch.cat([visual_input, point_position], dim=1)
            selected_joint_names += ['distractor']

        return visual_input, selected_joint_names


    def get_AT_data_gestalten(self, asf_path, amc_path, frame_samples):
        # self._num_dimensions = 3
        visual_input, selected_joint_names = self.compile_data(asf_path=asf_path, amc_path=amc_path, frame_samples=frame_samples)
        visual_input = visual_input.permute(1,0,2)
        visual_input = self.std_scale_data(visual_input, 15)        

        if self.distractor:
            point_position = self.get_distractor_position(frame_samples)
            visual_input = torch.cat([visual_input, point_position], dim=1)
            selected_joint_names += ['distractor']

        if self._num_dimensions == 6:
            visual_input = self.get_gestalt_vel(visual_input, frame_samples)
        elif self._num_dimensions == 7:
            visual_input = self.get_gestalt_dir_mag(visual_input, frame_samples)


        # self._num_dimensions = 7

        return visual_input, selected_joint_names
    

    def convert_data_AT_to_LSTM(self, data):
        return data.reshape(1, self._num_dimensions*self._num_features)
        # return data.reshape(1, num_samples, self._num_dimensions*self._num_features)


    def convert_data_LSTM_to_AT(self, data): 
        return data.reshape(self._num_features, self._num_dimensions)
        # return data.reshape(num_samples, self._num_features, self._num_dimensions)


    def get_distractor_position(self, num_frames):
        x_turn = 1
        x_speed = 0.01
        x_radius = -0.3

        y_turn = 1
        y_speed = 0.01
        y_radius = -0.2

        z_turn = 1
        z_speed = 0.001
        z_radius = 0.2

        pos = torch.zeros(num_frames, 3)
        
        x_i = np.arange(-1*x_turn, 1*x_turn, x_speed)
        x = 0
        y_i = np.arange(-1*y_turn, 1*y_turn, y_speed)
        y = 0
        z_i = np.arange(-1*z_turn, 1*z_turn, z_speed)
        z = 0

        for frame in range(num_frames):
            
            pos[frame, 0] = x_i[x]
            if x == len(x_i)-1:
                x_turn *= -1
                x_i = np.arange(-1, 1, x_speed) *x_turn
                x = 0
            else: 
                x += 1
            
            pos[frame, 1] = y_i[y]
            if y == len(y_i)-1:
                y_turn *= -1
                y_i = np.arange(-1, 1, y_speed) *y_turn
                y = 0
            else: 
                y += 1

            pos[frame, 2] = z_i[z]
            if z == len(z_i)-1:
                z_turn *= -1
                z_i = np.arange(-1, 1, z_speed) * z_turn
                z = 0
            else: 
                z += 1

        pos = torch.mul(torch.acos(pos), torch.Tensor([x_radius, y_radius, z_radius]))
        pos = pos.reshape(num_frames, 1, 3)

        
        print('Created distractor.')

        return pos
            
                


        





