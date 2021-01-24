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

        # ALTERNATIVE: calculating gradients with respect to alphas -> update alphas -> neg: invers matmul                          
        # # initialize rotation angles 
        # self.alpha = Variable(torch.tensor(alpha_init), requires_grad=self.rotation_gradient_init)
        # self.beta = Variable(torch.tensor(beta_init), requires_grad=self.rotation_gradient_init)
        # self.gamma = Variable(torch.tensor(gamma_init), requires_grad=self.rotation_gradient_init)

        # # initialize dimensional rotation matrices 
        # self.R_x = Variable(torch.tensor([[1.0,0.0,0.0],
        #                         [0.0, torch.cos(self.alpha), - torch.sin(self.alpha)], 
        #                         [0.0, torch.sin(self.alpha), torch.cos(self.alpha)]]), 
        #                         requires_grad=self.rotation_gradient_init) 
        # self.R_y = Variable(torch.tensor([[torch.cos(self.beta), 0.0, torch.sin(self.beta)], 
        #                         [0.0,1.0,0.0],
        #                         [- torch.sin(self.beta), 0.0, torch.cos(self.beta)]]), 
        #                         requires_grad=self.rotation_gradient_init) 
        # self.R_z = Variable(torch.tensor([[torch.cos(self.gamma), - torch.sin(self.gamma), 0.0], 
        #                         [torch.sin(self.gamma), torch.cos(self.gamma), 0.0], 
        #                         [0.0,0.0,1.0]]), 
        #                         requires_grad=self.rotation_gradient_init)

        # # initialize rotation matrix
        # self.rotation_matrix = Variable(torch.matmul(self.R_z, torch.matmul(self.R_y, self.R_x)),
        #                                 requires_grad=self.rotation_gradient_init)


    def translation_bias_(self):
        return self.translation_bias


    def update_translation_bias_(self, gradient, learning_rate):
        self.translation_bias = Variable(self.translation_bias - learning_rate * gradient,
                                        requires_grad=self.translation_gradient_init)
        # print('Updated translation bias.')
        # print(self.translation_bias)
        return self.translation_bias


    def update_rotation_matrix_(self, gradient, learning_rate):
        self.rotation_matrix = Variable(self.rotation_matrix - learning_rate * gradient,
                                        requires_grad=self.rotation_gradient_init)
        # print('Updated rotation matrix.')
        # print(self.rotation_matrix)
        return self.rotation_matrix


    def rotation_matrix_(self):
        return self.rotation_matrix


    def rotate(self, input):
        return torch.matmul(self.rotation_matrix, input.T).T


    def rotate(self, input, rotation_matrix):
        return torch.matmul(rotation_matrix, input.T).T


    def translate(self,input): 
        return input + self.translation_bias


    def translate(self,input, translation_bias): 
        return input + translation_bias
