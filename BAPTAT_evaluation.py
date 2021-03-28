from os import TMP_MAX
import torch 
import matplotlib.pyplot as plt
import numpy as np

class BAPTAT_evaluator():

    def __init__(self, 
                 num_frames=None,
                 num_observations=None,
                 num_features=15, 
                 preprocessor=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_frames = num_frames
        if num_observations is None: 
            self.num_observations = num_features
        else:
            self.num_observations = num_observations
        self.num_features = num_features
        self.preprocessor = preprocessor

    def prediction_errors(self, 
                          observations, 
                          final_predictions, 
                          loss_function):

        prediction_error = []
        for i in range(self.num_frames-1):
            with torch.no_grad():
                obs_t = self.preprocessor.convert_data_AT_to_LSTM(observations[i+1]).to(self.device)
                pred_t = final_predictions[i].to(self.device)
                loss = loss_function(pred_t, obs_t[0])
                prediction_error.append(loss)

        return prediction_error


    def plot_prediction_errors(self, prediction_error):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(prediction_error, 'r')
        axes.grid(True)
        axes.set_xlabel('frames')
        axes.set_ylabel('prediction error')
        axes.set_title('Prediction error after active tuning')
        # plt.show()
        return fig


    def prediction_errors_nxm(self, 
                          observations, 
                          additional_features,
                          num_observed_features,
                          final_predictions, 
                          loss_function):

        prediction_error = []
        for i in range(self.num_frames-1):
            with torch.no_grad():
                obs = observations[i+1]
                obs = [obs[i] for i in range(num_observed_features) if (i not in additional_features)]
                obs_t = self.preprocessor.convert_data_AT_to_LSTM(torch.stack(obs)).to(self.device)
                pred_t = final_predictions[i].to(self.device)
                loss = loss_function(pred_t, obs_t[0])
                prediction_error.append(loss)

        return prediction_error


    def plot_at_losses(self, losses, title): 
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(losses, 'r')
        axes.grid(True)
        axes.set_xlabel('active tuning runs')
        axes.set_ylabel('loss')
        axes.set_title(title)
        # plt.show()
        return fig


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

    
    def plot_binding_matrix(self, binding_matrix=None, feature_names=None, title=None, observ_order=None): 
        if observ_order is None: 
            observ_order = range(self.num_observations)
        
        bm = binding_matrix.detach().numpy()
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111)
        cax = ax.matshow(bm)            # draws matrix
        cb = fig.colorbar(cax, ax=ax, shrink=0.7)   # draws colorbar

        ## Adds numbers to plot
        for (i, j), z in np.ndenumerate(bm): 
            # ndenumerate function for generating multidimensional index iterator.
            # NOTE i is y-coordinate (row) and j is x-coordinate (column)
            ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
            # adds a text into the plot where i and j are the coordinates
            # and z is the assigned number 

        ## adding titles
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in observ_order])
        ax.set_xlabel('observed input')
        ax.xaxis.set_label_position('top') 
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_ylabel('input feature')

        plt.title(title, size = 12, fontweight='bold')
        # plt.show()
        return fig

    
    def plot_binding_matrix_nxm(self, 
        binding_matrix, 
        feature_names, 
        num_observed_features, 
        additional_features, 
        title, 
        observ_order): 

        if observ_order is None: 
            observ_order = range(self.num_observations)

        bm = binding_matrix.detach().numpy()
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111)
        cax = ax.matshow(bm)                        # draws matrix
        cb = fig.colorbar(cax, ax=ax, shrink=0.7)   # draws colorbar

        ## Adds numbers to plot
        for (i, j), z in np.ndenumerate(bm): 
            # ndenumerate function for generating multidimensional index iterator.
            ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
            # NOTE i is y-coordinate and j is x-coordinate
            # adds a text into the plot where i and j are the coordinates
            # and z is the assigned number 

        ## adding titles
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names)
        ax.set_xlabel('observed input')
        ax.xaxis.set_label_position('top') 
        feature_names = [feature_names[i] for i in range(num_observed_features) if (i not in additional_features)]
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_ylabel('input feature')

        plt.title(title, size = 12, fontweight='bold')
        # plt.show()
        return fig


    def plot_outcast_gradients(self, oc_grads, feature_names, num_observed_features, additional_features, title): 
        oc = torch.stack(oc_grads)
        add_feature_grads = [oc[:, i] for i in range(num_observed_features) if (i in additional_features)]
        input_feature_grads = [oc[:, i] for i in range(num_observed_features) if (i not in additional_features)]
        
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        for grad in add_feature_grads:
            axes.plot(grad, 'r')

        for grad in input_feature_grads:
            axes.plot(grad, 'b')

        axes.grid(True)
        axes.set_xlabel('active tuning runs')
        axes.set_ylabel('gradients for entries')
        axes.set_title(title)
        # plt.show()
        return fig


    
    def FBE(self, bm, ideal): 

        # Mahdi-version
        fbe = 0
        c = 0
        for j in range(self.num_features):
            c += bm[j,j]
            a = torch.square(bm[j,j]-ideal[j,j])
            b = 0
            for i in range(self.num_features):
                if i != j: 
                    b += torch.square(bm[j,i])
            fbe += torch.sqrt(a+b)

        # new FBE
        # fbe = 0
        # c = 0
        # maxima = torch.argmax(bm, dim=0)
        # for j in range(self.num_observations): 
        #     m = maxima[j]
        #     a = torch.square(bm[j, m] - ideal[j, m])
        #     b = 0
        #     for i in range(self.num_features):
        #         if i!=m:
        #             b += torch.square(bm[j, i])
        #     fbe += torch.sqrt(a+b)

        return fbe

    
    def FBE_nxm(self, bm, ideal, additional_features): 
        j = 0
        bm_sq = bm.clone().detach()
        bm_sq = bm_sq[:-1]
        for i in additional_features:
            i = i-j
            j += 1
            bm_1 = bm_sq[:,:i]
            bm_2 = bm_sq[:,i+1:]
            bm_sq = np.hstack([bm_1, bm_2])

        bm_sq = torch.Tensor(bm_sq).to(self.device)
        
        fbe = self.FBE(
            bm_sq, 
            torch.Tensor(np.identity(self.num_features)).to(self.device))
        
        for j in additional_features:
            a = torch.square(bm[self.num_features,j]-ideal[self.num_features,j])
            b = 0
            for i in range(self.num_features):
                b += torch.square(bm[i,j])
            fbe += torch.sqrt(a+b)
        return fbe


    def clear_nxm_binding_matrix(self, bm, additional_features):
        j = 0
        bm_sq = bm.clone().detach()
        bm_sq = bm_sq[:-1]
        for i in additional_features:
            i = i-j
            j += 1
            bm_1 = bm_sq[:,:i]
            bm_2 = bm_sq[:,i+1:]
            bm_sq = np.hstack([bm_1, bm_2])
        
        bm_sq = torch.Tensor(bm_sq).to(self.device)
        return bm_sq


    def FBE_nxm_additional_features(self, bm, ideal, additional_features):
        fbe = 0
        oc_fbe = 0
        for j in range(self.num_observations):
            if j in additional_features:
                a = torch.square(bm[self.num_features,j]-ideal[self.num_features,j])
                b = 0
                for i in range(self.num_features):
                    b += torch.square(bm[i,j])
                fbe += torch.sqrt(a+b)
            else: 
                oc_fbe += torch.square(bm[-1,j])
        fbe += torch.sqrt(oc_fbe)
        
        return fbe





