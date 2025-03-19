import torch
import torch.nn as nn
import numpy as np


# if we have noise vector we need to change it to tensor which will more suitable for RNA such as (length of RNA sequence, embedding size)
# i use for this linear layer (H = W * noise + b) where W is weight matrix and b is bias vector
class NoiseToRNAEmbedding(nn.Module):
    def __init__(self, noise_dim, sequence_length, embedding_size):
        super(NoiseToRNAEmbedding, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.linear = nn.Linear(noise_dim, sequence_length * embedding_size)

    def forward(self, noise):
        batch_size = noise.shape[0]
        h = self.linear(noise)
        return h.view(batch_size, self.sequence_length, self.embedding_size) * np.sqrt(self.embedding_size)


# now i have to change this tensor to sensor with positional encoding 
# H' =  h + PE where PE is positional encoding
# I use for this sinusoidal positional encoding but you can use trained positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, embedding_size):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        
        positions = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * -(np.log(10000.0) / embedding_size))

        pe = torch.zeros(seq_len, embedding_size)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('positional_encoding', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.positional_encoding


