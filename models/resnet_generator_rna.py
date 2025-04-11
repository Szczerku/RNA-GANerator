import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import NoiseToRNAEmbedding

"""
resnet_generator_rna.py
This module defines the ResNet-based generator architecture used in the WGAN-GP training.
The generator is designed to produce RNA sequences from random noise vectors.
The architecture consists of a series of residual blocks, each containing convolutional layers, batch normalization, and dropout for regularization.
The generator takes a latent noise vector as input and outputs a sequence of probabilities for the four nucleotides (A, C, G, U).
"""



class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = F.gelu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.gelu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return out + residual  # Skip connection
    

class ResNetGenerator(nn.Module):
    def __init__(self, latent_dim, sequence_length, embed_dim=256, n_blocks=4):
        super().__init__()
        self.sequence_length = sequence_length
        # shape: (batch_size, sequence_length, embed_dim)
        self.noise_embedding = NoiseToRNAEmbedding(latent_dim, sequence_length, embed_dim)

        # Example kernel sizes for residual blocks
        kernel_sizes = [3, 5, 7, 9]
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(embed_dim, kernel_size=k) for k in kernel_sizes]
        )
        # kernel_sizes = [3, 3, 5, 5, 7, 7, 9, 9]
        # self.res_blocks = nn.Sequential(
        #     *[ResidualBlock(embed_dim, kernel_size=k) for k in kernel_sizes]
        # )


        self.out = nn.Conv1d(embed_dim, 4, kernel_size=1) 

    def forward(self, noise):
        # Noise embedding
        embedded = self.noise_embedding(noise)
        # Reshape to (batch_size, embed_dim, sequence_length)
        embedded = embedded.permute(0, 2, 1)
        # Residual blocks
        out = self.res_blocks(embedded)
        # Output layer
        out = self.out(out)
        # Reshape to (batch_size, sequence_length, 4)
        out = out.permute(0, 2, 1)
        probs = F.gumbel_softmax(out, tau=0.5, hard=True)
        return probs
    

# test 
# noise = torch.randn(32, 128)  # Example noise vector
# generator = ResNetGenerator(latent_dim=128, sequence_length=109)
# output = generator(noise)
# print(output.shape)  # Should be (32, 109, 4)