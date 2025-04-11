import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class DiscriminatorRNA(nn.Module):
    def __init__(self, sequence_length, hidden_size=128, num_layers=1, dropout=0.5):
        super(DiscriminatorRNA, self).__init__()

        self.sequence_length = sequence_length
        input_size = 4  # 4 nukleotydy

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

        # Osłabiona warstwa w pełni połączona
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size // 2, 1))
        )

    def forward(self, x):
        # Dodajemy losowy szum, by zmniejszyć dokładność modelu
        x = x + 0.05 * torch.randn_like(x)

        x = x.permute(0, 2, 1)  # [batch_size, 4, sequence_length]
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, hidden_size]

        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)  # Uśrednianie zamiast max pooling

        x = self.fc_layers(x)
        x = torch.sigmoid(x)
        return x
