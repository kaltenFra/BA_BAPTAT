import torch 
from torch import nn
import matplotlib.pyplot as plt

# import sys
# sys.path.insert(0, 'D:\Uni\Kogni\Bachelorarbeit\Code\CoreLSTM\core_lstm.py')

from CoreLSTM.core_lstm import CORE_NET
from CoreLSTM.test_core_lstm import LSTM_Tester
from Data_Compiler.data_preparation import Preprocessor

class LSTM_Trainer():
    ## General parameters 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, loss_function=nn.MSELoss(), learning_rate=0.0001):
        self._model = CORE_NET()
        self._loss_function = loss_function
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        print('Initialized model!')
        print(self._model)
        print(self._loss_function)
        print(self._optimizer)


    def train(self, epochs, train_sequence, save_path):
        losses = []

        for i in range(epochs):
            for seq, labels in train_sequence:
                self._optimizer.zero_grad()

                y_pred, state = self._model(seq, state)

                single_loss = self._loss_function(y_pred, labels)
                single_loss.backward()
                self._optimizer.step()

            # save loss of epoch
            losses.append(single_loss.item())
            if i%25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        
        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
        
        self.save_model(save_path)
        self.plot_losses(losses)

        return losses
    

    def plot_losses(self, losses):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(losses, 'r')
        axes.grid(True)
        axes.set_xlabel('epochs')
        axes.set_ylabel('prediction error')
        axes.set_title('History of MSELoss during training')
        plt.show()


    def save_model(self, path):
        torch.save(self._model.state_dict(), path)
        print('Model was saved in: ' + path)

    

def main():
    # LSTM parameters
    frame_samples = 1000
    train_window = 10
    testing_size = 100

    # Init tools
    prepro = Preprocessor(15, 3)
    trainer = LSTM_Trainer()
    tester = LSTM_Tester()

    # Init tools
    data_asf_path = 'Data_Compiler/S35T07.asf'
    data_amc_path = 'Data_Compiler/S35T07.amc'
    model_save_path = 'CoreLSTM/models/LSTM_1.pt'

    # Preprocess data
    io_seq, dt_train, dt_test = prepro.get_LSTM_data(data_asf_path, 
                                                    data_amc_path, 
                                                    frame_samples, 
                                                    testing_size, 
                                                    train_window)

    # Train LSTM
    trainer.train(100, io_seq, model_save_path)

    test_input = dt_train[0,-train_window:]
    


if __name__ == "__main__":
    main()

