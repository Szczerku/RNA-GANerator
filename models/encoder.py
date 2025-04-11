import torch
import torch.nn as nn
import math

"""
Implementation of the Transformer encoder for RNA sequence generation
"""

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape = (batch_size, seq_len, d_model) 
        # -> (batch_size, seq_len, d_ff)  ->
        # -> (batch_size, seq_len, d_model) 
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model # embedding size for each nucleotid
        self.num_heads = num_heads # number of heads in multi-head attention
        self.d_k = d_model // num_heads # dimension of each head
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model) # output layer

        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1) # (batch_size, num_heads, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return torch.matmul(attention_scores, value), attention_scores
    

    def forward(self, q, k, v, mask):
        # q.shape = (batch_size, seq_len, d_model)
        # k.shape = (batch_size, seq_len, d_model)
        # v.shape = (batch_size, seq_len, d_model)
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) 
        # -> (batch_size, num_heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x, _ = self.attention(query, key, value, mask, self.dropout)

        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k)
        # -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)
        x = self.W_o(x)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, d_model, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection1 = ResidualConnection(d_model, dropout)
        self.residual_connection2 = ResidualConnection(d_model, dropout)
    
    def forward(self, x, mask):
        x = self.residual_connection1(x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connection2(x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
