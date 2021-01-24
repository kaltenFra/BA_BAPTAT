import torch
from torch.autograd import Variable

class Binder():
    """
    Performs Binding task. 
    """

    def __init__(self, num_features, gradient_init):
        self.gradient_init = gradient_init
        self.num_features = num_features
        bm = torch.rand(num_features, num_features)
        # make matrix stochastic
        sums = torch.sum(bm, dim=1)
        for i in range(self.num_features):
            bm[i] = bm[i]/sums[i]
        
        self.binding_matrix = Variable(bm, requires_grad=self.gradient_init)
        # self.binding_matrix = torch.nn.init.uniform_(
        #                         torch.empty(num_features, num_features, 
        #                                     requires_grad=self.gradient_init))
        # # make matrix stochastic
        # sums = torch.sum(self.binding_matrix, dim=1)
        # for i in range(self.num_features):
        #     self.binding_matrix[i] = self.binding_matrix[i]/sums[i]


    def binding_matrix_(self):
        return self.binding_matrix


    def update_binding_matrix_(self, gradient, learning_rate):
        bm = self.binding_matrix - learning_rate * gradient
        # make matrix stochastic
        sums = torch.sum(bm, dim=1)
        for i in range(self.num_features):
            bm[i] = bm[i]/sums[i]
        
        self.binding_matrix = Variable(bm, requires_grad=self.gradient_init)

        # self.binding_matrix = Variable(self.binding_matrix - learning_rate * gradient, 
        #                                 requires_grad=self.gradient_init)
        # # make matrix stochastic
        # sums = torch.sum(self.binding_matrix, dim=1)
        # for i in range(self.num_features):
        #     self.binding_matrix[i] = self.binding_matrix[i]/sums[i]
        
        # print('Updated binding matrix.')
        # print(self.binding_matrix)
        return self.binding_matrix


    def bind(self, input):
        binded = torch.matmul(self.binding_matrix, input)
        return binded


    def bind(self, input, binding_matrix):
        binded = torch.matmul(binding_matrix, input)
        # print(f'binded : {binded}')
        return binded
