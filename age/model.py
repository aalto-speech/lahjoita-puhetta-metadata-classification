import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models
from utils.tdnn import TDNN
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

import numpy as np
import math


class XVectorModel(nn.Module):
    def __init__(self, input_size, output_size, n_classes):
        super(XVectorModel, self).__init__()
       

        self.tdnn_1 = TDNN(input_size, output_size, context_size=5, dilation=1)
        self.tdnn_2 = TDNN(output_size, output_size, context_size=3, dilation=2)
        self.tdnn_3 = TDNN(output_size, output_size, context_size=3, dilation=3)
        self.tdnn_4 = TDNN(output_size, output_size, context_size=1, dilation=1)
        self.tdnn_5 = TDNN(output_size, 1500, context_size=1, dilation=1)
        self.lin_1 = nn.Linear(3000, output_size)
        self.lin_2 = nn.Linear(output_size, n_classes)


    def forward(self, input_tensor):
        input_tensor = input_tensor.permute(1, 0, 2)

        input_tensor = self.tdnn_1(input_tensor) 
        input_tensor = self.tdnn_2(input_tensor)
        input_tensor = self.tdnn_3(input_tensor)
        input_tensor = self.tdnn_4(input_tensor)
        input_tensor = self.tdnn_5(input_tensor)
        
        std, mean = torch.std_mean(input_tensor, unbiased=False, dim=1)

        output = torch.cat((std, mean), dim=1)
        output = self.lin_1(output)
        output = self.lin_2(output)
        
        return output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(0.1)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += self.pe[:x.size(0), :]
        return self.dropout(x)



class Encoder(nn.Module):
    def __init__(self, input_size, n_head_encoder, d_model, n_layers_encoder, max_len):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.n_head_encoder = n_head_encoder
        self.d_model = d_model
        self.n_layers_encoder = n_layers_encoder
        self.max_len = max_len

        # define the layers
        self.lin_transform = nn.Linear(self.input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_len)
        encoder_layers = TransformerEncoderLayer(self.d_model, self.n_head_encoder, self.d_model*4, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.n_layers_encoder)


    def forward(self, input_tensor, mask, src_key_padding_mask):
        input_tensor = self.lin_transform(input_tensor)
        input_tensor *= math.sqrt(self.d_model)
        input_encoded = self.pos_encoder(input_tensor)
        output = self.transformer_encoder(input_encoded, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output




class Transformer(nn.Module):
    def __init__(self, input_size, n_tokens, n_head_encoder, d_model, n_layers_encoder, max_len):
        super(Transformer, self).__init__()
         
        self.encoder = Encoder(input_size, n_head_encoder, d_model, n_layers_encoder, max_len)
        self.blstm = nn.LSTM(d_model,
                            d_model,
                            num_layers=2,
                            bidirectional=True
                            )


        self.lin_1 = nn.Linear(2*d_model, d_model)
        self.out = nn.Linear(d_model*2, n_tokens)
   

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    
    def generate_subsequent_mask(self, seq):
        len_s = seq.size(0)
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask


    def create_mask(self, src):
        src_seq_len = src.shape[0]

        src_mask = self.generate_square_subsequent_mask(src_seq_len).to(device=src.device)

        src_padding_mask = (src == 0).transpose(0, 1)
        return src_mask, src_padding_mask[:, :, 0]


    def forward(self, input_seq, x_vector_model):
        src_mask, src_padding_mask = self.create_mask(input_seq)
       
        # process the  audio
        memory = self.encoder(input_seq, mask=src_mask, src_key_padding_mask=src_padding_mask)
        output, hidden = self.blstm(memory)
        hidden = torch.mean(hidden[0], dim=0)

        x_vector_feat = x_vector_model(input_seq)
        x_vector_feat = self.lin_1(x_vector_feat)
        output = torch.cat((x_vector_feat, hidden), dim=1)
        output = self.out(output)

        return output
