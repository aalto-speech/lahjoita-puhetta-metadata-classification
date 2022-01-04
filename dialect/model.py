import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models
from utils.tdnn import TDNN

import numpy as np
import math


class XVectorModel(nn.Module):
    def __init__(self, input_size, output_size, n_classes, use_trn):
        super(XVectorModel, self).__init__()
        
        self.use_trn = use_trn
        self.output_size = output_size

        self.tdnn_1 = TDNN(input_size, output_size, context_size=5, dilation=1)
        self.tdnn_2 = TDNN(output_size, output_size, context_size=3, dilation=2)
        self.tdnn_3 = TDNN(output_size, output_size, context_size=3, dilation=3)
        self.tdnn_4 = TDNN(output_size, output_size, context_size=1, dilation=1)
        self.tdnn_5 = TDNN(output_size, 1500, context_size=1, dilation=1)
        self.lin_1 = nn.Linear(3000, output_size)
        
        if use_trn == True:
            self.blstm = nn.LSTM(768,
                            output_size,
                            num_layers=2,
                            bidirectional=True
                            ) 
            self.out = nn.Linear(output_size*2, n_classes)
        else:
            self.out = nn.Linear(output_size, n_classes)

   


    def forward(self, input_tensor, use_trn, trn_seqs=None):
        input_tensor = input_tensor.permute(1, 0, 2)

        input_tensor = self.tdnn_1(input_tensor) 
        input_tensor = self.tdnn_2(input_tensor)
        input_tensor = self.tdnn_3(input_tensor)
        input_tensor = self.tdnn_4(input_tensor)
        input_tensor = self.tdnn_5(input_tensor)
        
        std, mean = torch.std_mean(input_tensor, unbiased=False, dim=1)

        output = torch.cat((std, mean), dim=1)
        audio_embedding = self.lin_1(output)

        if use_trn == True:
            lstm_out, lstm_hidden = self.blstm(trn_seqs)
            trn_out = torch.mean(lstm_hidden[0], dim=0)
            output = torch.cat((audio_embedding, trn_out), dim=-1)
            output = self.out(output)
            return output
        else:
            output = self.out(audio_embedding)
            return output



