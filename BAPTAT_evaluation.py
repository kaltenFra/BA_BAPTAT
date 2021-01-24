import torch 
import matplotlib.pyplot as plt

class BAPTAT_evaluator():

    def __init__(self, 
                 num_frames, 
                 preprocessor):
        
        self.num_frames = num_frames
        self.preprocessor = preprocessor

    def prediction_errors(self, 
                          observations, 
                          final_predictions, 
                          loss_function):

        prediction_error = []
        for i in range(self.num_frames-1):
            with torch.no_grad():
                obs_t = self.preprocessor.convert_data_AT_to_LSTM(observations[i+1])
                pred_t = final_predictions[i]
                loss = loss_function(pred_t, obs_t)
                prediction_error.append(loss)

        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(prediction_error, 'r')
        axes.grid(True)
        axes.set_xlabel('frames')
        axes.set_ylabel('prediction error')
        axes.set_title('Prediction error after active tuning')
        plt.show()

        return prediction_error


    def help_visualize_devel(self, observations,final_predictions):
        at_final_pred_plot = final_predictions.reshape(self.num_frames, 15, 3)

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter3D(observations[0,:,0], 
                     observations[0,:,1], 
                     observations[0,:,2])
        ax.scatter3D(at_final_pred_plot[0,:,0], 
                     at_final_pred_plot[0,:,1], 
                     at_final_pred_plot[0,:,2])
        plt.show()

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter3D(observations[self.num_frames-1,:,0], 
                     observations[self.num_frames-1,:,1], 
                     observations[self.num_frames-1,:,2])
        ax.scatter3D(at_final_pred_plot[self.num_frames-1,:,0], 
                     at_final_pred_plot[self.num_frames-1,:,1], 
                     at_final_pred_plot[self.num_frames-1,:,2])
        plt.show()



