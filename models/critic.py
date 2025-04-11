import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


"""
critic.py

This module defines the Critic model architecture used in WGAN-GP training.  
The model combines a Convolutional Neural Network (CNN) for local feature extraction  
with a Bidirectional Long Short-Term Memory (BiLSTM) network for capturing sequential dependencies.  

The architecture is intentionally kept lightweight to ensure stable adversarial training—  
an overly strong Critic may hinder the Generator's ability to receive meaningful gradients.
"""


class Critic(nn.Module):
    def __init__(self, sequence_length, hidden_size=128, num_layers=1, dropout=0.5):
        super(Critic, self).__init__()            
        
        
        self.sequence_length = sequence_length
        input_size = 4 # 4 because of one-hot encoding (A, C, G, U)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_size // 2, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),  # NOWA
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size ,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size // 2, 1))
        )
    
    def forward(self, x): # x is tensor [batch_size, sequence_length, 4]
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1) 
        x = self.fc_layers(x)
        return x
