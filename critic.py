import torch
import torch.nn as nn
from torchinfo import summary
from torch.nn.utils import spectral_norm

class Critic(nn.Module):
    def __init__(self, sequence_length, hidden_size=64, num_layers=1, dropout=0.4):
        super(Critic, self).__init__()
        
        
        self.sequence_length = sequence_length
        input_size = 4 # 4 nukleotydy

        # Convolutional layers do wyodrębnienia cech z sekwencji
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=hidden_size // 2, out_channels= hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
        )

        # BILSTM do analizy sekwencji
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size ,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected layer do klasyfikacji
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            spectral_norm(nn.Linear(hidden_size, hidden_size)),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size, 1))
        )
    
    def forward(self, x):
        # x is tensor [batch_size, sequence_length, 4]
        # zmiana kształtu tensora na [batch_size, 4, sequence_length]
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.fc_layers(x)
        return x
    
# critic = Critic(sequence_length=100)  # Twój model
# summary(critic, input_size=(32, 100, 4))