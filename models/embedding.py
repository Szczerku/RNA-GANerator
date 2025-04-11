import torch
import torch.nn as nn
import numpy as np

"""
This module projects a random noise vector into a structured tensor suitable 
for further processing as an RNA-like sequence representation.
Specifically, the input noise vector is transformed using a linear layer 
into a tensor of shape (sequence_length, embedding_size), mimicking the 
structure of an embedded RNA sequence.
The output tensor is then reshaped to (batch_size, sequence_length, embedding_size)
"""

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


"""
This module applies sinusoidal positional encoding to a sequence tensor.
The encoding injects information about token positions within the sequence,
allowing models to distinguish between different positions.
This implementation follows the sinusoidal positional encoding approach 
introduced in "Attention is All You Need" (Vaswani et al., 2017).
"""
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


