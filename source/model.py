import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import dims
from easydict import EasyDict

model_hparams = EasyDict(n_tokens = dims.interval + dims.velocity + dims.note_on + dims.note_off + dims.pedal_on + dims.pedal_off,
                         embedding_dim = 512,
                         hidden_dim = 1024
                        )

class Model(nn.Module):
    def __init__(self, model_hparams):
        super().__init__()
        self.hp = model_hparams
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.embedding = nn.Embedding(self.hp.n_tokens, self.hp.embedding_dim)
        self.rnn = nn.LSTM(input_size=self.hp.embedding_dim, hidden_size=self.hp.hidden_dim,
                        num_layers=3, batch_first=True, dropout=0.1)
        self.out_layer = nn.Linear(self.hp.hidden_dim, self.hp.n_tokens)
        
    def forward(self, x):
        # x : (batch, length)
        
        # (batch, length, model_dim)
        x = self.embedding(x)
        # (batch, length, hidden_dim)
        x, _ = self.rnn(x)
        # (batch, length, n_tokens)
        x = self.out_layer(x)
        return x
        