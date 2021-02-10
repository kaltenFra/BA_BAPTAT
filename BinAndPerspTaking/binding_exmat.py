import torch
from torch.autograd import Variable

class BinderExMat():
    """
    Performs Binding task. 
    """

    def __init__(self, num_features, gradient_init):
        self.gradient_init = gradient_init
        self.num_features = num_features


    def init_binding_matrix_(self):
        binding_matrix = Variable(torch.rand(self.num_features, self.num_features), 
                                    requires_grad=False)
        return binding_matrix


    def init_binding_entries_(self):
        binding_entries = torch.rand((self.num_features, self.num_features), 
                                      requires_grad=False)
        return binding_entries
    

    def compute_binding_matrix(self, entries, scaler):
        bes = []
        for i in range(self.num_features):
            bes.append(torch.stack(entries[i]))
        bm = scaler(torch.stack(bes))
        return bm


    def update_binding_matrix_(self, matrix, gradient, learning_rate):
        binding_matrix = Variable(matrix - learning_rate * gradient, 
                                  requires_grad=self.gradient_init)
        return binding_matrix

    
    def update_binding_entries_(self, entires, gradient, learning_rate):
        upd_entries = []
        for j in range(self.num_features):
            row = []
            for k in range(self.num_features):
                entry = entires[j][k] - learning_rate * gradient[j][k]
                row.append(entry)
            upd_entries.append(row)
        return upd_entries
        


    def bind(self, input, bind_matrix):
        binded = torch.matmul(bind_matrix, input)
        # binded = Variable(torch.matmul(bind_matrix, input), requires_grad=True)
        return binded

