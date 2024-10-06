from DecoderLayer import DecoderLayer
import torch.nn as nn
from PositionalEncodings import PositionalEncoding
import torch

class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, n_heads, num_layers, ff_hidden_dim, max_len, dropout=0.01):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.positional_encodings = PositionalEncoding(dim, max_len)
        self.decoder = nn.ModuleList([
             DecoderLayer(dim, n_heads, ff_hidden_dim, dropout) for _ in range(num_layers)   
        ])
        self.linear = nn.Linear(dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_seq):
        x = self.embedding(input_seq)
        x = self.positional_encodings(x)
        for layer in decoder:
            x = layer(x)
        logits = self.linear(x)
        return self.softmax(logits)