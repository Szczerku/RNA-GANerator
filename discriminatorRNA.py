import torch
import torch.nn as nn

class discriminatorRNA(nn.Module):
    def __init__(self, sequence_length, hidden_size=64, num_layers=1, dropout=0.4):
        super(discriminatorRNA, self).__init__()

        self.sequence_length = sequence_length
        input_size = 4  # 4 nukleotydy

        # Osłabione warstwy konwolucyjne
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=128, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )

        # BiLSTM do analizy sekwencji (mniejszy hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Osłabiona warstwa w pełni połączona
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # Mniej neuronów
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
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
        return x
