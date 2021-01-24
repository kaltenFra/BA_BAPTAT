import torch
from torch import nn

class CORE_NET(nn.Module):
    def __init__(self, input_size=45, hidden_layer_size=90, num_layers=1, output_size=45):
        super(CORE_NET,self).__init__()
        self.hidden_size = hidden_layer_size
        self.hidden_num = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_layer_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bias=True)
        self.linear = nn.Linear(in_features=hidden_layer_size, out_features=output_size)

    def forward(self, input_seq, state=None):
        h0 = torch.zeros(self.hidden_num, input_seq.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.hidden_num, input_seq.size(0), self.hidden_size).requires_grad_()

        lstm_out, (hn, cn) = self.lstm(input_seq.reshape(len(input_seq) ,1, -1), (h0.detach(), c0.detach()))
        # lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1], (hn, cn)