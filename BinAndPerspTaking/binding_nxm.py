import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.serialization import PROTOCOL_VERSION

class BINDER_NxM():
    """
    Performs Binding task. 
    """

    def __init__(self, num_observations, num_features, gradient_init):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_init = gradient_init
        self.num_features = num_features
        self.num_observations = num_observations
        self.nxm = (num_features != num_observations)
        self.bin_momentum = [None, None]


    def init_binding_matrix_rand_(self):
        binding_matrix = torch.rand(self.num_features, self.num_observations)
        return binding_matrix


    def init_binding_matrix_det_(self):
        init_val = 1.0/self.num_features
        binding_matrix = torch.Tensor(self.num_features, self.num_observations)
        binding_matrix.requires_grad = False
        binding_matrix = binding_matrix.fill_(init_val)
        return binding_matrix


    def ideal_nxm_binding(self, additional_features, ideal_matrix):
        zeros = np.zeros((self.num_features, 1))
        for i in additional_features: 
            ideal_1 = ideal_matrix[:, :i]
            ideal_2 = ideal_matrix[:, i:]
            ideal_matrix = np.hstack([ideal_1, zeros, ideal_2])
        
        dummy_line = np.zeros((1, self.num_observations))
        for i in additional_features: 
            dummy_line[0, i] = 1
        
        ideal_matrix = np.vstack([ideal_matrix, dummy_line])
        
        return torch.Tensor(ideal_matrix)
    

    def scale_binding_matrix(self, 
        bm=None, 
        scale_mode='rcwSM', 
        scale_combo='comp_mult', 
        nxm_enhance = 'square', 
        nxm_last_line_scale = 0.1):

        if scale_mode == 'sigmoid':
            # compute sigmoidal
            return nn.functional.sigmoid(bm)
        elif scale_mode == 'rwSM': 
            return nn.functional.softmax(bm, dim=1)
        elif scale_mode == 'cwSM': 
            return nn.functional.softmax(bm, dim=0)
        elif scale_mode == 'rcwSM': 

            ## compute seperately
            if self.nxm: 
                # rowwise softmax
                bmrw = nn.functional.softmax(bm[:-1], dim=1)
                bmrw = torch.cat([bmrw, torch.ones(1, self.num_observations)])

                # columnwise softmax with modified 
                bm_last = bm[-1]
                s = torch.sign(bm_last)
                # alternatives for enhancing outcast line
                if nxm_enhance == 'square':
                    bm_last = torch.mul(s, torch.square(bm_last))
                elif nxm_enhance == 'squareroot' :
                    bm_last = torch.mul(s, torch.sqrt(torch.mul(s,bm_last)))
                elif nxm_enhance == 'log10' :
                    bm_last = torch.log10(9*bm_last + 1)
                else:
                    if scale_mode != None:
                        print('ERROR: Unknown enhancement! Skipped.')
                
                bm_last = nxm_last_line_scale * bm_last
                
                bm = torch.cat([bm[:-1], bm_last.view(1, self.num_observations)])
                bmcw = nn.functional.softmax(bm, dim=0)

            else:           
                # rowwise Softmax
                bmrw = nn.functional.softmax(bm, dim=1)

                # columnwise Softmax
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


    def decay_update_binding_matrix_(self, matrix, gradient, learning_rate, momentum):
        mom = momentum*self.calc_momentum()
        decay = self.get_decay(matrix)
        lmd = 0.001
        binding_matrix = matrix - learning_rate * gradient - torch.mul(decay, matrix) - torch.mul(decay, gradient) + mom
        # binding_matrix = matrix - learning_rate * gradient - decay + mom
        # binding_matrix = matrix - learning_rate * gradient - torch.mul(decay, matrix) + mom
        # binding_matrix = matrix - learning_rate * gradient - torch.div(decay, gradient) + mom
        # binding_matrix = matrix - learning_rate * gradient - torch.mul(decay, matrix) + torch.mul(decay, gradient) + mom
        # binding_matrix = matrix - learning_rate * gradient - torch.mul(decay, matrix) + torch.mul(lmd*decay, gradient) + mom
        # binding_matrix = matrix - learning_rate * gradient - torch.div(torch.mul(decay, matrix), gradient) + mom
        # binding_matrix = matrix - learning_rate * gradient - torch.mul(decay, torch.div(matrix, gradient)) + mom

        # binding_matrix = self.clampBM(binding_matrix)

        return binding_matrix

        
    def update_momentum(self, entries): 
        if self.bin_momentum[0] == None: 
            self.bin_momentum[0] = torch.Tensor(entries.copy()).to(self.device)
            self.bin_momentum[1] = torch.Tensor(entries.copy()).to(self.device)
        else: 
            self.bin_momentum[1] = self.bin_momentum[0]
            self.bin_momentum[0] = torch.Tensor(entries.copy()).to(self.device)

    
    def calc_momentum(self):
        if self.bin_momentum[0] == None: 
            if self.nxm: 
                return torch.zeros(self.num_features+1, self.num_observations).to(self.device)
            else:
                return torch.zeros(self.num_features, self.num_features).to(self.device)
        else:
            return self.bin_momentum[0] - self.bin_momentum[1]


    def bind(self, input, bind_matrix):
        binded = torch.matmul(bind_matrix, input)
        return binded


    def clampBM(self, bm): 
        # m = torch.median(bm)
        m = torch.mean(bm)
        std = torch.std(bm)
        return torch.clamp(bm, min=m-3*std, max=m+3*std)


    def get_decay(self, bm): 
        lmd = 0.001
        l2 = torch.sum(torch.mul(bm, bm))
        # print(l2)
        # lambda_b = torch.sigmoid(bm/l2) * lmd
        # lambda_b = torch.sigmoid(bm/(l2*lmd))
        # lambda_b = torch.sigmoid(bm/(l2*lmd)) * lmd
        # lambda_b = torch.sigmoid(l2*bm) * lmd
        lambda_b = torch.sigmoid(bm) * lmd * l2

        sig_bm = torch.sigmoid(bm)
        l2_sig = torch.sum(torch.mul(sig_bm, sig_bm))
        lambda_b = sig_bm * lmd * l2_sig

        # print(lambda_b)

        return lambda_b




