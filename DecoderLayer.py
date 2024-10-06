import torch.nn as nn
import torch

class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, ff_hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, dim)
        )
        self.layer_1_norm = nn.LayerNorm(dim)
        self.layer_2_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        attn_output, _  = self.attn(x, x, x)
        x = self.layer_1_norm(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_2_norm(x + self.dropout(ff_output))
        return x
        