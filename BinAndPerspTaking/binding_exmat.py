import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.serialization import PROTOCOL_VERSION

class BinderExMat():
    """
    Performs Binding task. 
    """

    def __init__(self, num_features, gradient_init):
        self.gradient_init = gradient_init
        self.num_features = num_features
        self.num_observations = num_features
        self.bin_momentum = [None, None]


    def init_binding_matrix_rand_(self):
        # binding_matrix = torch.rand(self.num_features, self.num_features)
        binding_matrix = torch.rand(self.num_observations, self.num_features)
        return binding_matrix


    def init_binding_matrix_det_(self):
        init_val = 1.0/self.num_features
        # binding_matrix = torch.Tensor(self.num_features, self.num_features)
        # binding_matrix = torch.Tensor(self.num_observations, self.num_features)
        binding_matrix = torch.Tensor(self.num_features, self.num_observations)
        # binding_matrix = torch.Tensor(self.num_features+1, self.num_observations)
        binding_matrix.requires_grad = False
        binding_matrix = binding_matrix.fill_(init_val)
        return binding_matrix


    def init_binding_entries_rand_(self):
        binding_entries = torch.rand((self.num_features, self.num_features), 
                                      requires_grad=False)
        return binding_entries
    
    def init_binding_entries_det_(self):
        init_val = 1.0/self.num_features
        binding_entries = torch.Tensor(self.num_features, self.num_features)
        binding_entries.requires_grad = False
        binding_entries = binding_entries.fill_(init_val)
        return binding_entries


    def ideal_nxm_binding(self, additional_features):
        ideal = np.identity(self.num_features)
        zeros = np.zeros((self.num_features, 1))
        for i in additional_features: 
            ideal_1 = ideal[:, :i]
            ideal_2 = ideal[:, i:]
            ideal = np.hstack([ideal_1, zeros, ideal_2])
        
        dummy_line = np.zeros((1, self.num_observations))
        for i in additional_features: 
            dummy_line[0, i] = 1
        
        ideal = np.vstack([ideal, dummy_line])
        
        return torch.Tensor(ideal)
    

    def compute_binding_matrix(self, entries):
        bes = []
        for i in range(self.num_features):
            bes.append(torch.stack(entries[i]))
        bm = torch.stack(bes)
        
        # compute sigmoidal
        # bm = nn.functional.sigmoid(bm)

        ## compute seperately
        bmrw = nn.functional.softmax(bm, dim=1)
        bmcw = nn.functional.softmax(bm, dim=0)
        
        # componentwise multiplication
        bm = torch.sqrt(torch.mul(bmrw, bmcw))
        # bm = torch.mul(bmrw, bmcw)
        # bm = torch.mean(torch.stack([bmrw, bmcw]), 0)

        # compute nested
        # bm = nn.functional.softmax(bmrw, dim=0)
        # bm = nn.functional.softmax(bmcw, dim=1)
       
        return bm

    def scale_binding_matrix(self, bm=None, scale_mode='rcwSM', scale_combo='comp_mult'):

        if scale_mode == 'sigmoid':
            # compute sigmoidal
            return nn.functional.sigmoid(bm)
        elif scale_mode == 'rwSM': 
            return nn.functional.softmax(bm, dim=1)
        elif scale_mode == 'cwSM': 
            return nn.functional.softmax(bm, dim=0)
        elif scale_mode == 'rcwSM': 
            ## compute seperately
            bmrw = nn.functional.softmax(bm, dim=1)
            bmcw = nn.functional.softmax(bm, dim=0)

            if scale_combo == 'comp_mult':
                # componentwise multiplication
                return torch.sqrt(torch.mul(bmrw, bmcw))
            elif scale_combo == 'comp_mean':
                # componentwise mean
                return torch.mean(torch.stack([bmrw, bmcw]), 0)
            elif scale_combo == 'nested_rw(cw)':
                return nn.functional.softmax(bmcw, dim=1)
            elif scale_combo == 'nested_cw(rw)':
                return nn.functional.softmax(bmrw, dim=0)
            else: 
                print('ERROR: Unknown combination! \nChoose valid combination.')
                exit()
        else: 
            if scale_mode != 'unscaled': 
                print('ERROR: Unknown mode! Return unscaled binding matrix.')
            return bm
                 


    def update_binding_matrix_(self, matrix, gradient, learning_rate, momentum):
        mom = momentum*self.calc_momentum()
        binding_matrix = matrix - learning_rate * gradient + mom
        return binding_matrix

    def update_binding_matrix_nxm_(self, matrix, gradient, learning_rate, momentum):
        mom = momentum*self.calc_momentum_nxm()
        binding_matrix = matrix - learning_rate * gradient + mom
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

    def calc_momentum_nxm(self):
        if self.bin_momentum[0] == None: 
            return torch.zeros(self.num_features+1, self.num_observations)
        else:
            return self.bin_momentum[1] - self.bin_momentum[0]

    def bind(self, input, bind_matrix):
        binded = torch.matmul(bind_matrix, input)
        return binded


class BinderExMat_nxm():
    """
    Performs Binding task. 
    """

    def __init__(self, num_features, num_observations, gradient_init):
        self.gradient_init = gradient_init
        self.num_features = num_features
        self.num_observations = num_observations
        self.bin_momentum = [None, None]


    def init_binding_matrix_rand_(self):
        # binding_matrix = torch.rand(self.num_features, self.num_features)
        binding_matrix = torch.rand(self.num_observations, self.num_features)
        return binding_matrix


    def init_binding_matrix_det_(self):
        init_val = 1.0/self.num_features
        # binding_matrix = torch.Tensor(self.num_features, self.num_features)
        # binding_matrix = torch.Tensor(self.num_observations, self.num_features)
        binding_matrix = torch.Tensor(self.num_features, self.num_observations)
        # binding_matrix = torch.Tensor(self.num_features+1, self.num_observations)
        binding_matrix.requires_grad = False
        binding_matrix = binding_matrix.fill_(init_val)
        return binding_matrix


    def init_binding_entries_rand_(self):
        binding_entries = torch.rand((self.num_features, self.num_features), 
                                      requires_grad=False)
        return binding_entries
    
    def init_binding_entries_det_(self):
        init_val = 1.0/self.num_features
        binding_entries = torch.Tensor(self.num_features, self.num_features)
        binding_entries.requires_grad = False
        binding_entries = binding_entries.fill_(init_val)
        return binding_entries


    def ideal_nxm_binding(self, additional_features):
        ideal = np.identity(self.num_features)
        zeros = np.zeros((self.num_features, 1))
        for i in additional_features: 
            ideal_1 = ideal[:, :i]
            ideal_2 = ideal[:, i:]
            ideal = np.hstack([ideal_1, zeros, ideal_2])
        
        dummy_line = np.zeros((1, self.num_observations))
        for i in additional_features: 
            dummy_line[0, i] = 1
        
        ideal = np.vstack([ideal, dummy_line])
        
        return torch.Tensor(ideal)
    

    def compute_binding_matrix(self, entries):
        bes = []
        for i in range(self.num_features):
            bes.append(torch.stack(entries[i]))
        bm = torch.stack(bes)
        
        # compute sigmoidal
        # bm = nn.functional.sigmoid(bm)

        ## compute seperately
        bmrw = nn.functional.softmax(bm, dim=1)
        bmcw = nn.functional.softmax(bm, dim=0)
        
        # componentwise multiplication
        bm = torch.sqrt(torch.mul(bmrw, bmcw))
        # bm = torch.mul(bmrw, bmcw)
        # bm = torch.mean(torch.stack([bmrw, bmcw]), 0)

        # compute nested
        # bm = nn.functional.softmax(bmrw, dim=0)
        # bm = nn.functional.softmax(bmcw, dim=1)
       
        return bm

    def scale_binding_matrix(self, bm):
        # bm = torch.cat([bm, torch.zeros(1,self.num_observations)])
              
        # compute sigmoidal
        # bm = nn.functional.sigmoid(bm)

        ## compute seperately
        bmrw = nn.functional.softmax(bm[:-1], dim=1)
        bmrw = torch.cat([bmrw, torch.ones(1, self.num_observations)])
        
        bm_last = bm[-1]
        s = torch.sign(bm_last)
        # alternatives for enhancing outcast line
        bm_last = torch.mul(s, torch.square(bm_last))
        # bm_last = torch.mul(s, torch.sqrt(torch.mul(s,bm_last)))
        # bm_last = torch.log10(9*bm_last + 1)
        # bm_last = 2 * bm_last
        # bm_last = 0.5 * bm_last
        bm_last = 0.1 * bm_last
        bm = torch.cat([bm[:-1], bm_last.view(1, self.num_observations)])
        bmcw = nn.functional.softmax(bm, dim=0)
        
        # componentwise combination
        bm = torch.sqrt(torch.mul(bmrw, bmcw))
        # bm = torch.mul(bmrw, bmcw)
        # bm = torch.mean(torch.stack([bmrw, bmcw]), 0)
       
        return bm


    def update_binding_matrix_(self, matrix, gradient, learning_rate, momentum):
        mom = momentum*self.calc_momentum()
        binding_matrix = matrix - learning_rate * gradient + mom
        return binding_matrix

    def update_binding_matrix_nxm_(self, matrix, gradient, learning_rate, momentum):
        mom = momentum*self.calc_momentum_nxm()
        binding_matrix = matrix - learning_rate * gradient + mom
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

    def calc_momentum_nxm(self):
        if self.bin_momentum[0] == None: 
            return torch.zeros(self.num_features+1, self.num_observations)
        else:
            return self.bin_momentum[1] - self.bin_momentum[0]

    def bind(self, input, bind_matrix):
        binded = torch.matmul(bind_matrix, input)
        return binded

