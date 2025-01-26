import random
import numpy as np
import torch
import torch.nn as nn
from FL_HELPERS.FL_constants import GENERAL_INFO
from Hyperparameters import Hyperparameters
from MODELS.AGCRN.AGCRN import AGCRN
from MODELS.ARIMA.ARIMA import ARIMA
from MODELS.LSTM_DSTGCRN.LSTM_DSTGCRN import LSTM_DSTGCRN
from MODELS.GRU.GRU import GRU
from MODELS.LR.LR import LR
from MODELS.LSTM.LSTM import LSTM 


class LocalModel():

    def __init__(self, params=None, verbose=True):

        self.params = params
        if self.params is None:
            # This is for the initialization for the socket
            self.params = Hyperparameters()
        
        if self.params.model_name=='Fed-LSTM-DSTGCRN':
            self.model = LSTM_DSTGCRN(self.params)

        elif self.params.model_name=='FedAGCRN':
            self.model = AGCRN(self.params)

        elif self.params.model_name=='FedLSTM':
            self.model = LSTM(self.params)

        elif self.params.model_name=='FedGRU': 
            self.model = GRU(self.params)

        elif self.params.model_name=='FedARIMA':
            self.model = ARIMA(self.params)

        elif self.params.model_name=='FedLR':
            self.model = LR(self.params)

        # Set seed for reproducibility
        torch.cuda.cudnn_enabled = False
        torch.backends.cudnn.deterministic = True
        if self.params.seed is not None:
            random.seed(self.params.seed)
            np.random.seed(self.params.seed)
            torch.manual_seed(self.params.seed)
            torch.cuda.manual_seed(self.params.seed)

        # Initialize weights
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        # Choose device and move the model to the device
        # Only for CPU, because the GPU is set in the FL_Client
        if not torch.cuda.is_available():
            self.params.device = "cpu"

        if 0:
            print(f"{GENERAL_INFO} {self.params.model_name} Using {self.params.device}")

        self.model = self.model.to(self.params.device)

                    
        
