import torch
from abc import ABC, abstractmethod
from BinAndPerspTaking.binding_nxm import BINDER_NxM
from BinAndPerspTaking.perspective_taking import Perspective_Taker
from CoreLSTM.core_lstm import CORE_NET
from Data_Compiler.data_preparation import Preprocessor
from BAPTAT_evaluation import BAPTAT_evaluator

class BAPTAT_BASICS(ABC): 

    def __init__(self, num_features, num_observations, num_dimensions): 
        self.num_features = num_features
        self.num_observations = num_observations
        self.num_dimensions = num_dimensions

        print('Initialized BAPTAT basic module.')


    def set_data_parameters_(self, 
        num_frames, 
        num_observations, 
        num_input_features, 
        num_input_dimesions): 

        ## Define data parameters
        self.num_frames = num_frames
        self.num_observations = num_observations
        self.num_input_features = num_input_features
        self.num_input_dimensions = num_input_dimesions
        self.input_per_frame = self.num_input_features * self.num_input_dimensions
        self.nxm = (self.num_observations != self.num_input_features)
        
        self.binder = BINDER_NxM(
            num_observations=self.num_observations, 
            num_features=self.num_input_features, 
            gradient_init=True)

        self.perspective_taker = Perspective_Taker(
            self.num_input_features, 
            self.num_input_dimensions,
            rotation_gradient_init=True, 
            translation_gradient_init=True)

        self.preprocessor = Preprocessor(
            self.num_observations, 
            self.num_input_features, 
            self.num_input_dimensions)

        self.evaluator = BAPTAT_evaluator(
            self.num_frames, 
            self.num_observations, 
            self.num_input_features, 
            self.preprocessor)


    def init_model_(self, model_path): 
        ## Load model
        self.core_model = CORE_NET()
        self.core_model.load_state_dict(torch.load(model_path))
        self.core_model.eval()

        print('Model loaded.')


    def init_general_inference_tools(self):
        # general
        self.obs_count = 0
        self.at_inputs = torch.tensor([]).to(self.device)
        self.at_predictions = torch.tensor([]).to(self.device)
        self.at_final_predictions = torch.tensor([]).to(self.device)
        self.at_losses = []

        # state 
        self.at_states = []


    def init_binding_inference_tools(self):
        self.Bs = []
        self.B_grads = [None] * (self.tuning_length+1)
        self.B_upd = [None] * (self.tuning_length+1)
        self.bm_losses = []
        self.bm_dets = []
        self.oc_grads = []


    def init_rotation_inference_tools(self):
        self.Rs = []
        self.R_grads = [None] * (self.tuning_length+1)
        self.R_upd = [None] * (self.tuning_length+1)
        self.rm_losses = []
        self.rv_losses = []


    def init_translation_inference_tools(self):
        self.Cs = []
        self.C_grads = [None] * (self.tuning_length+1)
        self.C_upd = [None] * (self.tuning_length+1)
        self.c_losses = []


    def set_comparison_values_binding(self, ideal_binding): 
        self.ideal_binding = ideal_binding.to(self.device)
        if self.nxm:
            self.ideal_binding = self.binder.ideal_nxm_binding(
                self.additional_features, self.ideal_binding).to(self.device)


    def set_comparison_values(self, ideal_rotation_values, ideal_rotation_matrix):
        self.ideal_rotation = ideal_rotation_matrix
        if self.rotation_type == 'qrotate': 
            self.ideal_quat = ideal_rotation_values
            self.ideal_angle = self.perspective_taker.qeuler(self.ideal_quat, 'xyz')
        elif self.rotation_type == 'eulrotate': 
            self.ideal_angle = ideal_rotation_values
        else: 
            print(f'ERROR: Received unknown rotation type!\n\trotation type: {self.rotation_type}')
            exit()

    
    def set_comparison_values_translation(self, ideal_translation):
        self.ideal_translation = ideal_translation


    def perform_binding(self, data, bm_activations):
        bm = self.binder.scale_binding_matrix(
            bm_activations, 
            self.scale_mode, 
            self.scale_combo, 
            self.nxm_enhance, 
            self.nxm_last_line_scale)
        if self.nxm:
            bm = bm[:-1]

        return self.binder.bind(data, bm)


    def perform_rotation(self, data, rotation_vals):
        if self.rotation_type == 'qrotate': 
            return self.perspective_taker.qrotate(data, rotation_vals)
        else:
            rotmat = self.perspective_taker.compute_rotation_matrix_(
                rotation_vals[0], rotation_vals[1], rotation_vals[2])
            return self.perspective_taker.rotate(data, rotmat)


    def perform_translation(self, data, translation_vals):
        return self.perspective_taker.translate(data, translation_vals)


        