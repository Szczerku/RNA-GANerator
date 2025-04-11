import torch.nn as nn
import torch.nn.functional as F


from .embedding import NoiseToRNAEmbedding, PositionalEncoding
from .encoder import Encoder, EncoderBlock, MultiHeadAttention, FeedForward

"""
Generator RNA
Parameters:
    - latent_dim: int - dimension of the noise vector
    - sequence_length: int - length of the RNA sequence
    - d_model: int - dimension of the model
    - num_layers: int - number of transformer encoder layers
    - num_heads: int - number of heads in multi-head attention
    - d_ff: int - dimension of the feed-forward layer
    - lstm_hidden_size: int - dimension of the LSTM hidden state
    - lstm_layers: int - number of LSTM layers
"""

class GeneratorRNA(nn.Module):
    def __init__(self, latent_dim, sequence_length, d_model, num_layers, num_heads, d_ff, lstm_hidden_size=256, lstm_layers=2, dropout=0.1):
        super(GeneratorRNA, self).__init__()

        # Noise embedding and positional encoding functions
        self.noise_embedding = NoiseToRNAEmbedding(latent_dim, sequence_length, d_model)
        self.positional_encoding = PositionalEncoding(sequence_length, d_model)

        # Transformer encoder blocks
        encoder_blocks = []
        for _ in range(num_layers):
            attention_block = MultiHeadAttention(d_model, num_heads, dropout)
            feed_forward_block = FeedForward(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(attention_block, feed_forward_block, d_model, dropout)
            encoder_blocks.append(encoder_block)
        
        self.encoder = Encoder(d_model, encoder_blocks)

        self.lstm = nn.LSTM(
            input_size = d_model,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )

        self.output = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.LayerNorm(lstm_hidden_size // 2),
            nn.GELU(),
            nn.Linear(lstm_hidden_size // 2, 4)  # 4 nucleotides A, C, G, U
        )

    def forward(self, noise):
        # Noise embedding
        embedded = self.noise_embedding(noise)
        
        # Add positional encoding
        embedded_with_pos = self.positional_encoding(embedded)
        
        # Transformer processing
        transformer_output = self.encoder(embedded_with_pos, mask=None)
         
        # LSTM processing
        lstm_output, _ = self.lstm(transformer_output)
        
        # output layer
        logits = self.output(lstm_output)
        probs = F.gumbel_softmax(logits, tau=0.5, hard=True)
        return probs # probs - [batch, sequence_length, 4] - 4 nucleotides A, C, G, U each sequence = [[0,1,0,0], ... etc.