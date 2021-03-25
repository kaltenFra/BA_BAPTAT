import torch 
from torch import nn
import matplotlib.pyplot as plt
import math

import sys
sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')
from CoreLSTM.core_lstm import CORE_NET
from CoreLSTM.test_core_lstm_gestalten import LSTM_Tester
from Data_Compiler.data_preparation import Preprocessor
from torch.utils.data import TensorDataset, DataLoader


class LSTM_Trainer():
    ## General parameters 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, loss_function, learning_rate, momentum, l2_penality, batch_size, hidden_num):
        self._model = CORE_NET(input_size=105, hidden_layer_size=hidden_num)
        self.batch_size = batch_size
        self._loss_function = loss_function
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2_penality)
        # self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate)
        # self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        print('Initialized model!')
        print(self._model)
        print(self._loss_function)
        print(self._optimizer)


    def train(self, epochs, train_sequence, save_path):
        losses = []
        num_batches = len(train_sequence)
        seq_size = len(train_sequence[0][0])
        closed_loop = 0
        closed_loop_sizes = [2,5,10,20]
        # closed_loop_epsilon = 0.0005
        closed_loop_epsilon = 0.00000001
        # closed_loop_epsilon = 0.0000005
        prev_ep_loss = 100000000000
        # print(num_batches)
        # print(seq_size)
        for ep in range(epochs):
            self._model.zero_grad()
            self._optimizer.zero_grad()
            # ep_loss = 0

            #########################################################################################
            # Teacher Forcing
            # inputs = []
            # targets = []
            # for seq, labels in train_sequence:
            #     inputs.append(seq)
            #     # print(seq[1:,:].shape)
            #     # print(labels.shape)
            #     target = torch.cat((seq[1:,:], labels), dim=0)
            #     targets.append(target)
            
            # batch_size = seq.size()[0]
            # ins = []
            # tars = []
            # for i in range(len(train_sequence)):
            #     if i%batch_size==0:
            #         ins.append(inputs[i])
            #         tars.append(targets[i])
            # ins = torch.stack(ins)
            # tars = torch.stack(tars)
            # num_batches = ins.size()[0]
            # num_input = ins.size()[2]
            # # print(num_batches)
            # state = self._model.init_hidden(num_batches)
            # # print(ins.shape)
            # # print(tars.shape)
            
            # outs = []
            # for i in range(batch_size):
            #     input = ins[:,i,:].view(num_batches, num_input)
            #     # print(input.shape)
            #     out, state = self._model.forward(input, state)
            #     outs.append(out)

            # outs = torch.stack(outs)
            # single_loss = self._loss_function(outs, tars.permute(1,0,2))
            # single_loss.backward()
            # self._optimizer.step()
            # # print(single_loss.item())

            # ep_loss = single_loss 

            #########################################################################################
            # Perform plain teacher Forcing on batchsizes until epsilon thrshhold reached 
            # then slowly increase closed loop size to capture long term dependencies 

            if closed_loop==0:

                inputs = []
                targets = []
                for seq, labels in train_sequence:
                    inputs.append(seq)
                    # print(seq[1:,:].shape)
                    # print(labels.shape)
                    target = torch.cat((seq[1:,:], labels), dim=0)
                    targets.append(target)
                
                batch_size = seq.size()[0]
                # print(batch_size)
                ins = []
                tars = []
                for i in range(len(train_sequence)):
                    if i%batch_size==0:
                        ins.append(inputs[i])
                        tars.append(targets[i])
                ins = torch.stack(ins)
                tars = torch.stack(tars)
                num_batches = ins.size()[0]
                num_input = ins.size()[2]
                # print(num_batches)
                state = self._model.init_hidden(num_batches)
                # print(ins.shape)
                # print(tars.shape)
                
                outs = []
                for i in range(batch_size):
                    input = ins[:,i,:].view(num_batches, num_input)
                    # print(input.shape)
                    out, state = self._model.forward(input, state)
                    outs.append(out)

                outs = torch.stack(outs)
                single_loss = self._loss_function(outs, tars.permute(1,0,2))
                single_loss.backward()
                self._optimizer.step()
                # print(single_loss.item())

                # ep_loss = single_loss

            else:
                # closed_loop = 5
                inputs = []
                targets = []
                num_bat_per_seq = math.floor(seq_size/closed_loop)
                # print(num_bat_per_seq)

                for seq, labels in train_sequence:
                    target = torch.cat((seq, labels), dim=0)
                    # print(target.shape)
                    for i in range(num_bat_per_seq):
                        inputs.append(seq[closed_loop*i:closed_loop*(i+1),:])
                        targets.append(target[closed_loop*(i+1)])
                        # print(seq[closed_loop*i:closed_loop*(i+1),:].shape)
                        # print(target[closed_loop*(i+1)].shape)

                    # print(inputs[1][0]==targets[0])
                
                # print(len(targets))
                # print(len(inputs))
                batch_size = seq.size()[0]
                # print(batch_size)
                distinct_batches = False
                if distinct_batches: 
                    ins = []
                    tars = []
                    for i in range(len(train_sequence)):
                        if i%batch_size==0:
                            ins.append(inputs[i])
                            tars.append(targets[i])
                else: 
                    ins = inputs
                    tars = targets
                ins = torch.stack(ins)
                tars = torch.stack(tars)
                # print(ins[1][0]==tars[0])
                # print(ins.size())
                # print(tars.size())

                num_batches = ins.size()[0]
                num_input = ins.size()[2]
                # print(num_batches)
                # print(num_input)
                state = self._model.init_hidden(num_batches)
                # print(ins.shape)
                # print(tars.shape)
                
                outs = []
                input = ins[:,0,:].view(num_batches, num_input)
                for i in range(closed_loop):
                    # print(input.shape)
                    input, state = self._model.forward(input, state)
                    outs.append(input)

                # print(input.shape)
                outs = torch.stack(outs).permute(1,0,2)
                # print(outs.shape)
                
                single_loss = self._loss_function(input, tars)
                # print(single_loss.item())

                single_loss.backward()
                self._optimizer.step()
            
            with torch.no_grad():
                ep_loss = single_loss.clone().detach()
                # ep_loss = single_loss / num_batches
            # print(ep_loss)
            
            # print(prev_ep_loss)
            # print(ep_loss)
            # print(prev_ep_loss-ep_loss)
            # print(prev_ep_loss-ep_loss < closed_loop_epsilon)
            if len(closed_loop_sizes)>0 and abs(prev_ep_loss-ep_loss) < closed_loop_epsilon:
                closed_loop = closed_loop_sizes[0]
                print(f'New loop length: {closed_loop}')
                closed_loop_sizes = closed_loop_sizes[1:]
                closed_loop_epsilon *= 0.1
                print(f'New loop epsilon: {closed_loop_epsilon}')
                # print(closed_loop)
                # print(closed_loop_sizes)
                # exit()
            prev_ep_loss = ep_loss




            #########################################################################################
            # Update State nach jedem Batch mit End-Prediction, inkl. Teacher Forcing 
            # DISTINCT! but same batches every epoch! 
            # inputs = []
            # targets = []
            # i = 0
            # batch_size = train_sequence[0][0].size()[0]
            # for seq, labels in train_sequence:
            #     if i%batch_size==0:
            #         inputs.append(seq)
            #         targets.append(labels)
                
            #     i += 1
            # inputs = torch.stack(inputs)
            # num_batches = inputs.size()[0]
            # num_input = inputs.size()[2]
            # targets = torch.stack(targets).view(num_batches, num_input)

            # state = self._model.init_hidden(num_batches)

            # outs = []
            # for i in range(batch_size):
            #     input = inputs[:,i,:].view(num_batches, num_input)
            #     out, state = self._model.forward(input, state)
                
            # single_loss = self._loss_function(out, targets)
            # single_loss.backward()
            # self._optimizer.step()

            # ep_loss = single_loss / num_batches
            # print(ep_loss)
            # # print(foo)

            #########################################################################################
            # Update State nach jedem Batch mit End-Prediction, inkl. Teacher Forcing 
            # DISTINCT! but same batches every epoch! 
            # if ep == 0: 

            #     inputs = []
            #     targets = []
            #     batch_size = train_sequence[0][0].size()[0]
            #     for seq, labels in train_sequence:
            #         inputs.append(seq)
            #         targets.append(labels)
                    
            #     inputs = torch.stack(inputs)
            #     num_batches = inputs.size()[0]
            #     batch_length = inputs.size()[1]
            #     num_input = inputs.size()[2]
            #     targets = torch.stack(targets)

            #     print(inputs.shape)
            #     print(targets.shape)
            #     train_loader = DataLoader(
            #         dataset=TensorDataset(inputs, targets),
            #         batch_size=5,
            #         pin_memory=True,
            #         shuffle=True
            #     )

            #     batches_per_epoch = len(train_loader)
            #     print(batches_per_epoch)
                
            #     print(batch_length)

            # for (ins, tars) in train_loader: 
            #     real_num_batches = ins.size()[0]

            #     self._optimizer.zero_grad()

            #     state = self._model.init_hidden(real_num_batches)
            #     outs = []
            #     for i in range(batch_length):
            #         input = ins[:,i,:].view(real_num_batches, num_input)
            #         out, state = self._model.forward(input, state)
            #         # outs.append(out)
            
            #     # outs = torch.stack(outs)
            #     # print(out.shape)
            #     # print(tars.view(real_num_batches, num_input).shape)
            #     single_loss = self._loss_function(out, tars.view(real_num_batches, num_input))
            #     single_loss.backward()
            #     self._optimizer.step()

            #     ep_loss += single_loss
            # print(ep_loss)
            # print(foo)


            #########################################################################################
            # Update State nach jedem Batch mit End-Prediction, inkl. Teacher Forcing 
            # NOT DISTINCT! 
            # for seq, labels in train_sequence:
            #     batch_size = seq.size()[0]
            #     self._optimizer.zero_grad()
            #     state = self._model.init_hidden(1)
            #     for s in seq:
            #         s = s.view(1,45)
            #         y_pred, state = self._model(s, state)
                
            #     single_loss = self._loss_function(y_pred, labels)
            #     single_loss.backward()
            #     self._optimizer.step()

            #     ep_loss += single_loss 

            #########################################################################################
            # Update State nach jedem Batch, kein Teacher Forcing
            # for seq, labels in train_sequence:
            #     batch_size = seq.size()[0]
            #     self._optimizer.zero_grad()
            #     state = self._model.init_hidden(batch_size)
            #     y_pred, state = self._model(seq, state)

            #     # print(foo)
            #     single_loss = self._loss_function(y_pred[-1], labels[0])
            #     single_loss.backward()
            #     self._optimizer.step()

            #     ep_loss += single_loss 
            
            # ep_loss /= num_batches

            # ep_loss /= batches_per_epoch

            # save loss of epoch
            losses.append(ep_loss.item())
            if ep%25 == 1:
                print(f'epoch: {ep:3} loss: {single_loss.item():10.8f}')
        
        print(f'epoch: {ep:3} loss: {single_loss.item():10.10f}')

        self.save_model(save_path)
        self.plot_losses(losses)

        return losses
    

    def plot_losses(self, losses):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(losses, 'r')
        axes.grid(True)
        axes.set_xlabel('epochs')
        axes.set_ylabel('loss')
        axes.set_title('History of MSELoss during training')
        plt.show()


    def save_model(self, path):
        torch.save(self._model.state_dict(), path)
        print('Model was saved in: ' + path)



    

def main():
    # LSTM parameters
    frame_samples = 1000
    train_window = 20
    testing_size = 100
    num_features = 15
    num_dimensions = 3
    hidden_num = 210

    # Training parameters
    epochs = 2000
    mse=nn.MSELoss()
    loss_function=nn.MSELoss()
    # loss_function= lambda x, y: mse(x, y) * (num_features * 7)
    # loss_function= lambda x, y: mse(x, y) * (num_features * 3)
    # loss_function=nn.L1Loss()
    # loss_function=nn.SmoothL1Loss()
    learning_rate=0.1
    momentum=0.1
    l2_penality=0.01

    # Init tools
    prepro = Preprocessor(num_features=num_features, num_dimensions=num_dimensions)
    trainer = LSTM_Trainer(
        loss_function, 
        learning_rate, 
        momentum, 
        l2_penality, 
        train_window, 
        hidden_num
    )
    tester = LSTM_Tester(loss_function)
    tester_1 = LSTM_Tester(mse)

    # Init tools
    data_asf_path = 'Data_Compiler/S35T07.asf'
    data_amc_path = 'Data_Compiler/S35T07.amc'
    model_save_path = 'CoreLSTM/models/LSTM_73_gestalten.pt'

    # Preprocess data
    io_seq, dt_train, dt_test = prepro.get_LSTM_data_gestalten(
        data_asf_path, 
        data_amc_path, 
        frame_samples, 
        testing_size, 
        train_window
    )

    # Train LSTM
    losses = trainer.train(epochs, io_seq, model_save_path)

    test_input = dt_train[0,-train_window:]

    # Test LSTM
    tester.test(testing_size, model_save_path, test_input, dt_test, train_window,hidden_num)
    tester_1.test(testing_size, model_save_path, test_input, dt_test, train_window, hidden_num)
    


if __name__ == "__main__":
    main()

