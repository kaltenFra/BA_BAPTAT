import torch
from torch import nn
import matplotlib.pyplot as plt

from CoreLSTM.core_lstm import CORE_NET

class LSTM_Tester(): 

    def __init__(self, loss_function=nn.MSELoss()):
        self._loss_function = loss_function


    def predict(self, num_predictions, model, test_input, test_target, train_window): 
        prediction_error = []

        for i in range(num_predictions):
            seq = test_input[-train_window:]

            with torch.no_grad():
                if i>0:
                    loss = self._loss_function(test_input[-1], test_target[0,i]).item()
                    prediction_error.append(loss)

                new_prediction, state = model(seq)
                test_input = torch.cat((test_input, new_prediction.reshape(1,45)), 0)

        predictions = test_input[-num_predictions:].reshape(num_predictions, 15, 3)

        return predictions, prediction_error

    
    def plot_pred_error(self, errors):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(errors, 'r')
        axes.grid(True)
        axes.set_xlabel('time steps')
        axes.set_ylabel('prediction error')
        axes.set_title('Prediction error during testing')
        plt.show()

    
    def test(self, num_predictions, model_path, test_input, test_target, train_window):
        model = CORE_NET()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(model)

        pred, pred_err = self.predict(num_predictions, model, test_input, test_target, train_window)

        self.plot_pred_error(pred_err)
