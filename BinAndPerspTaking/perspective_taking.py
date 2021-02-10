import torch 
from torch.autograd import Variable

class Perspective_Taker():

    """
    Performs Perspective Taking 
    """

    def __init__(self, alpha_init, beta_init, gamma_init, 
                rotation_gradient_init, translation_gradient_init):

        self.rotation_gradient_init = rotation_gradient_init
        self.translation_gradient_init = translation_gradient_init
        
        self.dimensions = 3
        
        ## Translation
        # initialize translation bias
        self.translation_bias = torch.rand(self.dimensions, requires_grad=self.translation_gradient_init)           

        ## Rotation
        # initialize rotation angles 
        self.alpha = Variable(torch.tensor(alpha_init), requires_grad=False)
        self.beta = Variable(torch.tensor(beta_init), requires_grad=False)
        self.gamma = Variable(torch.tensor(gamma_init), requires_grad=False)

        # initialize dimensional rotation matrices 
        self.R_x = Variable(torch.tensor([
            [1.0,0.0,0.0],
            [0.0, torch.cos(self.alpha), - torch.sin(self.alpha)], 
            [0.0, torch.sin(self.alpha), torch.cos(self.alpha)]]), 
            requires_grad=False) 
        self.R_y = Variable(torch.tensor([
            [torch.cos(self.beta), 0.0, torch.sin(self.beta)], 
            [0.0,1.0,0.0],
            [- torch.sin(self.beta), 0.0, torch.cos(self.beta)]]), 
            requires_grad=False) 
        self.R_z = Variable(torch.tensor([
            [torch.cos(self.gamma), - torch.sin(self.gamma), 0.0], 
            [torch.sin(self.gamma), torch.cos(self.gamma), 0.0], 
            [0.0,0.0,1.0]]), 
            requires_grad=False)

        # initialize rotation matrix
        self.rotation_matrix = Variable(torch.matmul(self.R_z, torch.matmul(self.R_y, self.R_x)),
                                        requires_grad=self.rotation_gradient_init)


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
        # initialize dimensional rotation matrices 
        R_x_1 = torch.Tensor([[1.0,0.0,0.0]])
        R_x_2 = torch.stack([torch.zeros(1), torch.cos(alpha), - torch.sin(alpha)], dim=1)
        R_x_3 = torch.stack([torch.zeros(1), torch.sin(alpha), torch.cos(alpha)], dim=1)
        R_x = torch.stack([R_x_1, R_x_2, R_x_3], dim=1)
        
        R_y_1 = torch.stack([torch.cos(beta), torch.zeros(1), torch.sin(beta)], dim=1)
        R_y_2 = torch.Tensor([[0.0,1.0,0.0]])
        R_y_3 = torch.stack([- torch.sin(beta), torch.zeros(1), torch.cos(beta)], dim=1)
        R_y = torch.stack([R_y_1, R_y_2, R_y_3], dim=1)

        R_z_1 = torch.stack([torch.cos(gamma), - torch.sin(gamma), torch.zeros(1)], dim=1)
        R_z_2 = torch.stack([torch.sin(gamma), torch.cos(gamma), torch.zeros(1)], dim=1)
        R_z_3 = torch.Tensor([[0.0,0.0,1.0]])
        R_z = torch.stack([R_z_1, R_z_2, R_z_3], dim=1)

        # initialize rotation matrix
        rotation_matrix = torch.matmul(R_z, torch.matmul(R_y, R_x))        
        return rotation_matrix


    def init_translation_bias_(self):
        tb = Variable(torch.rand(self.dimensions), requires_grad=self.translation_gradient_init )
        return tb

    def translation_bias_(self):
        return self.translation_bias


    def update_translation_bias_(self, gradient, learning_rate):
        self.translation_bias = Variable(self.translation_bias - learning_rate * gradient,
                                        requires_grad=self.translation_gradient_init)
        return self.translation_bias


    def update_translation_bias_(self, translation_bias, gradient, learning_rate):
        upd_translation_bias = Variable(translation_bias - learning_rate * gradient,
                                        requires_grad=self.translation_gradient_init)
        # print('Updated translation bias.')
        # print(self.translation_bias)
        return upd_translation_bias


    def update_rotation_angles_(self, rotation_angles, gradient, learning_rate):
        # update with gradient descent
        upd_rotation_angles = []
        for i in range(self.dimensions):
            with torch.no_grad():
                e = rotation_angles[i] - learning_rate * gradient[i]
                e = e % 360
                upd_rotation_angles.append(e)
        
        return upd_rotation_angles


    def update_rotation_matrix_(self, gradient, learning_rate):
        self.rotation_matrix = Variable(self.rotation_matrix - learning_rate * gradient,
                                        requires_grad=self.rotation_gradient_init)
        # print('Updated rotation matrix.')
        # print(self.rotation_matrix)
        return self.rotation_matrix
    

    def update_rotation_matrix_(self, rotation_matrix, gradient, learning_rate):
        # update with gradient descent
        rotation_matrix = Variable(rotation_matrix - learning_rate * gradient,
                                        requires_grad=self.rotation_gradient_init)
        
        return self.rotation_matrix


    def rotation_matrix_(self):
        return self.rotation_matrix


    def rotate(self, input, rotation_matrix):
        # return torch.matmul(rotation_matrix, input.T).T
        return torch.matmul(rotation_matrix, input.T).T


    def translate(self,input, translation_bias): 
        return input + translation_bias
