import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from tqdm import tqdm
import copy
import numpy as np
from numpy.linalg import norm

def cosine_similarity(v, w):
        # Implement the cosine similarity
        return np.dot(v,w) / (norm(v) * norm(w))

# RNN version as baseline
class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_RNN, self).__init__()

        # embed Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        #rnn layer from pytorch
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output
    
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()
    
    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens 
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]  
        top_scores = scores[indexes]
        return (indexes, top_scores)

# Replace RNN with LSTM and add Dropout layers 
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, with_dropout=False, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)
        self.with_dropout = with_dropout

        if with_dropout:
            self.out_dropout = nn.Dropout(out_dropout)
            self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        if self.with_dropout:
            emb = self.emb_dropout(emb)

        lstm_out, _  = self.lstm(emb)

        if self.with_dropout:
            lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)

        return output

    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)
                          
