import torch
from torch import nn

class CORE_NET(nn.Module):
    def __init__(self, input_size=45, hidden_layer_size=360):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(CORE_NET,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_layer_size
        self.lstm = nn.LSTMCell(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            bias=True)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.input_size)

    def forward(self, input_seq, state=None):
        if state==None:
            h0 = torch.zeros(input_seq.size(0), self.hidden_size).requires_grad_().to(self.device)
            c0 = torch.zeros(input_seq.size(0), self.hidden_size).requires_grad_().to(self.device)
            
        hn, cn = self.lstm(input_seq, state)
        prediction = self.linear(hn)
        return prediction, (hn,cn)

    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size).to(self.device),
                torch.zeros(batch_size, self.hidden_size).to(self.device))


class PSEUDO_CORE(): 
    def __init__(self, input_size=45):
        self.input_size = input_size
    
    def forward(self, input, binding_matrix, observation, div_obs):
        with torch.no_grad():
            div = observation - input
        o_t = input+div
        o_t1 = o_t+div_obs
        # print(o_t)
        # print(div_obs)
        print(div)
        # print(o_t1)
        # print(foo)
        return o_t1

# class CORE_NET_Layer(nn.Module):
#     def __init__(self, input_size=45, hidden_layer_size=360, num_layers=2, output_size=45):
#         super(CORE_NET,self).__init__()
#         self.hidden_size = hidden_layer_size
#         self.hidden_num = num_layers
#         self.lstm = nn.LSTM(
#             input_size=input_size, 
#             hidden_size=hidden_layer_size, 
#             num_layers=num_layers, 
#             batch_first=True, 
#             bias=True)
#         self.linear = nn.Linear(in_features=hidden_layer_size, out_features=output_size)

#     def forward(self, input_seq, state=None):
#         if state==None:
#             h0 = torch.zeros(self.hidden_num, input_seq.size(0), self.hidden_size).requires_grad_()
#             c0 = torch.zeros(self.hidden_num, input_seq.size(0), self.hidden_size).requires_grad_()
#             state = (h0.detach(), c0.detach())
            
#         lstm_out, (hn, cn) = self.lstm(input_seq.reshape(len(input_seq) ,1, -1), state)
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         return predictions[-1], (hn, cn)