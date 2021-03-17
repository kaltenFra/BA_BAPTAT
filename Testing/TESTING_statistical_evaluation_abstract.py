import pandas as pd   
import matplotlib.pyplot as plt
import abc 
from abc import ABC, abstractmethod

import torch 
import os
import sys
sys.path.append('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT')

class TEST_STATISTICS(ABC): 

    def __init__(self, num_features, num_observations, num_dimensions): 
        self.num_features = num_features
        self.num_observations = num_observations
        self.num_dimensions = num_dimensions

        print('Initialized statistical evaluator.')