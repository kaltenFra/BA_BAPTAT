import torch
from torch.autograd import Variable

class BinderExMat():
    """
    Performs Binding task. 
    """

    def __init__(self, num_features, gradient_init):
        self.gradient_init = gradient_init
        self.num_features = num_features
        self.bin_momentum = [None, None]


    def init_binding_matrix_(self):
        binding_matrix = Variable(torch.rand(self.num_features, self.num_features), 
                                    requires_grad=False)
        return binding_matrix


    def init_binding_entries_rand_(self):
        binding_entries = torch.rand((self.num_features, self.num_features), 
                                      requires_grad=False)
        return binding_entries
    
    def init_binding_entries_det_(self):
        init_val = 1.0/self.num_features
        binding_entries = torch.Tensor(self.num_features, self.num_features)
        binding_entries.requires_grad_ = False
        binding_entries = binding_entries.fill_(init_val)
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

    
    def update_binding_entries_(self, entries, gradient, learning_rate, momentum):
        upd_entries = []
        mom = momentum*self.calc_momentum()
        for j in range(self.num_features):
            row = []
            for k in range(self.num_features):
                # entry = entries[j][k] - learning_rate * gradient[j][k] 
                entry = entries[j][k] - learning_rate * gradient[j][k] + mom[j][k]
                row.append(entry)
            upd_entries.append(row)
        self.update_momentum(entries)
        return upd_entries
        
    def update_momentum(self, entries): 
        if self.bin_momentum[0] == None: 
            self.bin_momentum[0] = torch.Tensor(entries.copy())
            self.bin_momentum[1] = torch.Tensor(entries.copy())
        else: 
            self.bin_momentum[1] = self.bin_momentum[0]
            self.bin_momentum[0] = torch.Tensor(entries.copy())
    
    def calc_momentum(self):
        if self.bin_momentum[0] == None: 
            return torch.zeros(self.num_features, self.num_features)
        else:
            return self.bin_momentum[1] - self.bin_momentum[0]


    def bind(self, input, bind_matrix):
        binded = torch.matmul(bind_matrix, input)
        # binded = Variable(torch.matmul(bind_matrix, input), requires_grad=True)
        return binded

