import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict

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
    
    def _get_initial_state(self, batch_size):
        h = torch.zeros(3, batch_size, self.hp.hidden_dim).cuda()
        c = torch.zeros(3, batch_size, self.hp.hidden_dim).cuda()
        return (h, c)
    
    def inference(self, x, state=None, temperature=1.0):
        # x : (batch, length)
        
        # (batch, length, model_dim)
        x = self.embedding(x)
        # (batch, length, hidden_dim)
        x, state = self.rnn(x, state)
        # (batch, length, n_tokens)
        x = self.out_layer(x)
        # (batch, 1)
        x = torch.distributions.categorical.Categorical(logits=x[:, -1:]/temperature).sample()
        return x, state
        