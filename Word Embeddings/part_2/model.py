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

class VariationalDropout(nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if self.dropout <= 0.:
            return x

        variational = torch.bernoulli(torch.empty_like(x).fill_(1 - self.dropout))
        output = x * variational / (1 - self.dropout)
        return output

class LM_LSTM_Optimized(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, variational_dropout=False, weight_tying=False, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_Optimized, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

        if weight_tying:
            self.lstm = nn.LSTM(emb_size, emb_size, n_layers, bidirectional=False) # map the lstm output layer with the embedding layer
            self.output.weight = self.embedding.weight # share the weights with the embedding layer
        else:
            self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)

        if variational_dropout:
            self.out_dropout = VariationalDropout(out_dropout)
            self.emb_dropout = VariationalDropout(emb_dropout)
        else:
            self.out_dropout = nn.Dropout(out_dropout)
            self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)

        lstm_out, _  = self.lstm(emb)

        output = self.out_dropout(lstm_out)
        output = self.output(output).permute(0,2,1)

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