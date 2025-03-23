import torch
import torch.nn as nn

## UWAGA NIE UWZGLEDNIA PADDINGU Jeśli twoje RNA może mieć różne długości, 
# warto dodać pack_padded_sequence i pad_packed_sequence W FUNKCJI FORWARD!!!!
class discriminatorRNA(nn.Module):
    def __init__(self, sequence_length, hidden_size=256, num_layers=2, dropout=0.4):
        super(discriminatorRNA, self).__init__()
        
        self.sequence_length = sequence_length
        input_size = 4 # 4 nukleotydy

        # Convolutional layers do wyodrębnienia cech z sekwencji
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=256, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )

        # BILSTM do analizy sekwencji
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer do klasyfikacji
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
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


