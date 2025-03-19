import torch
import torch.nn as nn
import torch.nn.functional as F

from noise import generate_noise
from embedding import NoiseToRNAEmbedding, PositionalEncoding
from bert import Encoder, EncoderBlock, MultiHeadAttention, FeedForward

class generatorRNA(nn.Module):
    def __init__(self, latent_dim, sequence_length, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(generatorRNA, self).__init__()

        # noise embedding and positional encoding
        self.noise_embedding = NoiseToRNAEmbedding(latent_dim, sequence_length, d_model)
        self.positional_encoding = PositionalEncoding(sequence_length, d_model)

        # transformer encoder blocks
        encoder_blocks = []
        for _ in range(num_layers):
            attention_block = MultiHeadAttention(d_model, num_heads, dropout)
            feed_forward_block = FeedForward(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(attention_block, feed_forward_block, d_model, dropout)
            encoder_blocks.append(encoder_block)
        
        self.encoder = Encoder(d_model, encoder_blocks)

        self.output = nn.Linear(d_model, 4)

    def forward(self, noise):
        # noise embedding
        embedded = self.noise_embedding(noise)
        
        # add positional encoding
        embedded_with_pos = self.positional_encoding(embedded)
        
        # transformer processing
        transformer_output = self.encoder(embedded_with_pos, mask=None)
        
        # output projection to nucleotide space
        output = self.output(transformer_output)
        
        # Używamy softmax aby uzyskać prawdopodobieństwa dla każdego nukleotydu
        output_probs = F.softmax(output, dim=-1)
        
        # Konwersja do postaci one-hot (opcjonalnie)
        nucleotide_indices = torch.argmax(output_probs, dim=-1)
        one_hot_output = F.one_hot(nucleotide_indices, num_classes=4).float()
        
        return one_hot_output  # lub return output_probs, jeśli potrzebujesz prawdopodobieństw
        


# Test
latent_dim = 256 # its value that describes the size of the latent space 
batch_size = 32 # its the number of samples f.e if you have batch size 32 then sample size is 32

sequence_length = 100

num_layers = 12
num_heads = 8
d_model = 512
d_ff = 2048
dropout = 0.1

model = generatorRNA(latent_dim, sequence_length, d_model, num_layers, num_heads, d_ff, dropout)

noise = generate_noise(latent_dim, batch_size)

output = model(noise)

# Drukowanie wyników
print("Kształt wyjścia:", output.shape)  # Powinno być [batch_size, sequence_length, 4]
print("Przykładowe wektory one-hot:")
print(output[0])  # Pokazuje pierwszą sekwencję z batcha