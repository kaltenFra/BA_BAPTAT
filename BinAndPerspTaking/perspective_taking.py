import torch 
import numpy as np
from torch.autograd import Variable

class Perspective_Taker():

    """
    Performs Perspective Taking 
    """

    def __init__(self, num_features, num_dimensions, 
                rotation_gradient_init, translation_gradient_init):

        self.rotation_gradient_init = rotation_gradient_init
        self.translation_gradient_init = translation_gradient_init
        
        self.dimensions = num_dimensions
        self.num_features = num_features

        self.bin_momentum_rotation = [None, None]
        self.bin_momentum_translation = [None, None]


    #################################################################################
    #################### ROTATION
    #################################################################################

    def init_rotation_matrix_(self):
        # initialize dimensional rotation matrices 
        R_x = Variable(torch.tensor([
            [1.0,0.0,0.0],
            [0.0, torch.cos(self.alpha), - torch.sin(self.alpha)], 
            [0.0, torch.sin(self.alpha), torch.cos(self.alpha)]]), 
            requires_grad=False) 
        R_y = Variable(torch.tensor([
            [torch.cos(self.beta), 0.0, torch.sin(self.beta)], 
            [0.0,1.0,0.0],
            [- torch.sin(self.beta), 0.0, torch.cos(self.beta)]]), 
            requires_grad=False) 
        R_z = Variable(torch.tensor([
            [torch.cos(self.gamma), - torch.sin(self.gamma), 0.0], 
            [torch.sin(self.gamma), torch.cos(self.gamma), 0.0], 
            [0.0,0.0,1.0]]), 
            requires_grad=False)


        # initialize rotation matrix
        rotation_matrix = Variable(torch.matmul(R_z, torch.matmul(R_y, R_x)),
                                        requires_grad=self.rotation_gradient_init)
        
        return rotation_matrix


    # ALTERNATIVE: calculating gradients with respect to alphas -> update alphas -> neg: invers matmul                          
    def init_angles_(self):
        return torch.rand((self.dimensions, 1), requires_grad=False) * 360

    
    def compute_rotation_matrix_(self, alpha, beta, gamma):
        alpha_rad = torch.deg2rad(alpha)
        beta_rad = torch.deg2rad(beta)
        gamma_rad = torch.deg2rad(gamma)
        # initialize dimensional rotation matrices 
        R_x_1 = torch.Tensor([[1.0,0.0,0.0]])
        R_x_2 = torch.stack([torch.zeros(1), torch.cos(alpha_rad), - torch.sin(alpha_rad)], dim=1)
        R_x_3 = torch.stack([torch.zeros(1), torch.sin(alpha_rad), torch.cos(alpha_rad)], dim=1)
        R_x = torch.stack([R_x_1, R_x_2, R_x_3], dim=1)
        
        R_y_1 = torch.stack([torch.cos(beta_rad), torch.zeros(1), torch.sin(beta_rad)], dim=1)
        R_y_2 = torch.Tensor([[0.0,1.0,0.0]])
        R_y_3 = torch.stack([- torch.sin(beta_rad), torch.zeros(1), torch.cos(beta_rad)], dim=1)
        R_y = torch.stack([R_y_1, R_y_2, R_y_3], dim=1)

        R_z_1 = torch.stack([torch.cos(gamma_rad), - torch.sin(gamma_rad), torch.zeros(1)], dim=1)
        R_z_2 = torch.stack([torch.sin(gamma_rad), torch.cos(gamma_rad), torch.zeros(1)], dim=1)
        R_z_3 = torch.Tensor([[0.0,0.0,1.0]])
        R_z = torch.stack([R_z_1, R_z_2, R_z_3], dim=1)

        # initialize rotation matrix
        rotation_matrix = torch.matmul(R_z, torch.matmul(R_y, R_x))    
        return rotation_matrix


    def update_rotation_angles_(self, rotation_angles, gradient, learning_rate):
        # update with gradient descent
        upd_rotation_angles = []
        for i in range(self.dimensions):
            with torch.no_grad():
                e = rotation_angles[i] - learning_rate * gradient[i]
                e = e % 360
                upd_rotation_angles.append(e)
        
        return upd_rotation_angles


    def rotate(self, input, rotation_matrix):
        # return torch.matmul(rotation_matrix, input.T).T
        return torch.matmul(rotation_matrix, input.T).T

            
    def update_momentum_rotation(self, entries): 
        if self.bin_momentum_rotation[0] == None: 
            self.bin_momentum_rotation[0] = torch.Tensor(entries.clone().detach())
            self.bin_momentum_rotation[1] = torch.Tensor(entries.clone().detach())
        else: 
            self.bin_momentum_rotation[1] = self.bin_momentum_rotation[0]
            self.bin_momentum_rotation[0] = torch.Tensor(entries.clone().detach())
    
    def calc_momentum_translation(self):
        if self.bin_momentum_rotation[0] == None: 
            return torch.zeros(self.num_features, self.num_features)
        else:
            return self.bin_momentum_rotation[1] - self.bin_momentum_rotation[0]

    #################################################################################
    #################### TRANSLATION
    #################################################################################


    def init_translation_bias_(self):
        tb = torch.rand(self.dimensions)
        return tb


    def update_translation_bias_(self, translation_bias, gradient, learning_rate):
        upd_translation_bias = Variable(
            translation_bias - learning_rate * gradient,
            requires_grad=self.translation_gradient_init)

        return upd_translation_bias
    
    def update_translation_bias_(self, translation_bias, gradient, learning_rate, momentum):
        mom = momentum*self.calc_momentum_translation()
        
        upd_translation_bias = Variable(
            translation_bias - learning_rate * gradient + momentum,
            requires_grad=self.translation_gradient_init)
        # print('Updated translation bias.')

        self.update_momentum_translation(translation_bias)

        return upd_translation_bias


    def translate(self,input, translation_bias): 
        return input + translation_bias

    
    def update_momentum_translation(self, entries): 
        if self.bin_momentum_translation[0] == None: 
            self.bin_momentum_translation[0] = torch.Tensor(entries.clone().detach())
            self.bin_momentum_translation[1] = torch.Tensor(entries.clone().detach())
        else: 
            self.bin_momentum_translation[1] = self.bin_momentum_translation[0]
            self.bin_momentum_translation[0] = torch.Tensor(entries.clone().detach())
    
    def calc_momentum_translation(self):
        if self.bin_momentum_translation[0] == None: 
            return torch.zeros(self.num_features, self.num_features)
        else:
            return self.bin_momentum_translation[1] - self.bin_momentum_translation[0]


def main(): 
    perspective_taker = Perspective_Taker(15, 3, rotation_gradient_init=True, translation_gradient_init=True)
    
    rm = perspective_taker.compute_rotation_matrix_(torch.tensor([0.]), torch.tensor([90.]), torch.tensor([180.]))
    print(rm)

if __name__ == "__main__":
    main()