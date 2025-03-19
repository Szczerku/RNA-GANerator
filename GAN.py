import torch
import torch.nn as nn
from noise import generate_noise
from embedding import NoiseToRNAEmbedding, PositionalEncoding
from bert import Encoder, EncoderBlock, MultiHeadAttention, FeedForward, LayerNormalization
# generate noise for generator in agan network

latent_dim = 256 # its value that describes the size of the latent space 
sample_size = 32 # its the number of samples f.e if you have batch size 32 then sample size is 32

sequence_length = 100
embedding_size = 512

noise = generate_noise(latent_dim, sample_size)
print("Shape of noise: ", noise.shape)

# create instance of NoiseToTensor and pass noise vector to it
noise_to_rna = NoiseToRNAEmbedding(latent_dim, sequence_length, embedding_size)

# convert noise vector to tensor
tensor = noise_to_rna(noise) 
print(tensor.shape) # torch.Size([8, 16])

# test positional encoding
positional_encoding = PositionalEncoding(sequence_length, embedding_size)
tensor_with_positional_encoding = positional_encoding(tensor)

#print(tensor_with_positional_encoding)
#print(tensor_with_positional_encoding.shape) # [1. 8, 16] -> 1 is batch size, 8 is length of RNA sequence and 16 is embedding size


num_layers = 12
num_heads = 8
d_model = 512
d_ff = 2048
dropout = 0.1

# Tworzymy listę warstw enkodera
encoder_blocks = []
for _ in range(num_layers):
    attention_block = MultiHeadAttention(d_model, num_heads, dropout)
    feed_forward_block = FeedForward(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(attention_block, feed_forward_block, d_model, dropout)
    encoder_blocks.append(encoder_block)

# Tworzymy obiekt Encoder
encoder = Encoder(d_model, encoder_blocks)

output = encoder(tensor_with_positional_encoding, mask=None)
#print(output)
#print("enkoder shape", output.shape) # torch.Size([1, 8, 512])

#class RNAGenerator

seq = nn.Linear(512, 4)

prob_output = seq(output)
#print(prob_output)
#print(prob_output.shape) # torch.Size([1, 8, 4])

nucleotides = ['A', 'C', 'G', 'U']
max_values, max_indices = torch.max(prob_output, dim=-1)

# Wyświetlenie wyników dla całego batcha
for batch in range(32):
    rna_sequence = "".join(nucleotides[max_indices[batch, i].item()] for i in range(sequence_length))
    print(f"Batch {batch + 1}: {rna_sequence}") 

