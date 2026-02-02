import torch
import torch.nn as nn
import math

class CompressionTransformer(nn.Module):
    """
    Compresses a sequence of tokens into a fixed-size semantic vector.
    Uses a small Transformer Encoder architecture.
    """

    def __init__(self, vocab_size=120000, d_model=256, nhead=4, num_layers=2, output_dim=64, max_len=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.compressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        self.d_model = d_model

    def forward(self, x):
        """
        x: (batch, seq_len) - Token indices
        return: (batch, output_dim) - Semantic vector
        """
        # Ensure input is long tensor (indices)
        if x.dtype != torch.long:
            x = x.long()
            
        # Embed and add position encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Mean pooling over sequence length
        x = x.mean(dim=1)
        
        # Compress to output dimension
        return self.compressor(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1), :]
