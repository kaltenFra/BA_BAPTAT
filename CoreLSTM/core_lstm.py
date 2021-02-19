import torch
from torch import nn

class CORE_NET(nn.Module):
    def __init__(self, input_size=45, hidden_layer_size=360):
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
            h0 = torch.zeros(input_seq.size(0), self.hidden_size).requires_grad_()
            c0 = torch.zeros(input_seq.size(0), self.hidden_size).requires_grad_()
        #     state = (h0.detach(), c0.detach())
        # seq_len = len(input_seq)
        # input_seq = input_seq.view(seq_len ,self.input_size, -1)
        # print(input_seq.shape)
        # for i in range (seq_len):
        #     lstm_out, state = self.lstm(input_seq[i], state)

        hn, cn = self.lstm(input_seq, state)
        prediction = self.linear(hn)
        return prediction, (hn,cn)

    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size))

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