class TransformerLSTMGenerator(nn.Module):
    def __init__(self, latent_dim, sequence_length, d_model, num_layers, num_heads, d_ff, lstm_hidden_size=512, lstm_layers=2, dropout=0.1):
        super(TransformerLSTMGenerator, self).__init__()
        
        # Embedding szumu i kodowanie pozycyjne
        self.noise_embedding = NoiseToRNAEmbedding(latent_dim, sequence_length, d_model)
        self.positional_encoding = PositionalEncoding(sequence_length, d_model)
        
        # Bloki enkodera transformera
        encoder_blocks = []
        for _ in range(num_layers):
            attention_block = MultiHeadAttention(d_model, num_heads, dropout)
            feed_forward_block = FeedForward(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(attention_block, feed_forward_block, d_model, dropout)
            encoder_blocks.append(encoder_block)
        
        self.encoder = Encoder(d_model, encoder_blocks)
        
        # Warstwy LSTM przetwarzające output z transformera
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        
        # Projekcja wyjściowa do przestrzeni nukleotydów
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.LayerNorm(lstm_hidden_size // 2),
            nn.GELU(),
            nn.Linear(lstm_hidden_size // 2, 4)  # 4 nukleotydy: A, C, G, U
        )
        
    def forward(self, noise):
        # Embedding szumu
        embedded = self.noise_embedding(noise)
        
        # Dodaj kodowanie pozycyjne
        embedded_with_pos = self.positional_encoding(embedded)
        
        # Przetwarzanie przez transformer
        transformer_output = self.encoder(embedded_with_pos, mask=None)
        # transformer_output shape: [batch_size, sequence_length, d_model]
        
        # Przetwarzanie przez LSTM
        lstm_output, (hidden, cell) = self.lstm(transformer_output)
        # lstm_output shape: [batch_size, sequence_length, lstm_hidden_size]
        
        # Projekcja do przestrzeni nukleotydów
        logits = self.output_proj(lstm_output)
        # logits shape: [batch_size, sequence_length, 4]
        
        return logits
    
    def sample(self, noise, temperature=1.0):
        logits = self.forward(noise)
        probs = F.softmax(logits / temperature, dim=-1)
        nucleotide_indices = torch.multinomial(probs.view(-1, 4), 1).view(noise.size(0), -1)
        return nucleotide_indices