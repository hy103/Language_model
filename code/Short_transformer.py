from Multi_head_attention import Multihead_attention
from Skip_connections import GELU
from Layer_normalization import LayerNorm
import torch.nn as nn 
import math
import torch


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], cfg["emb_dim"]),
                GELU(),
                nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]),

        )

    def forward(self, x):
        return self.layers(x)

class Short_transformerblock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = Multihead_attention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_len= cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)
        self.layernorm1 = LayerNorm(cfg["emb_dim"])
        self.layernorm2 = LayerNorm(cfg["emb_dim"])
        self.drop_short = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):

        shortcut = x
        x = self.layernorm1(x)
        x = self.att(x)
        x = self.drop_short(x)
        x = x+shortcut

        shortcut = x
        x = self.layernorm2(x)
        x = self.att(x)
        x = self.drop_short(x)
        x = x+shortcut

        return x
    


GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

torch.manual_seed(123)
x = torch.rand(2, 4, 768)

st = Short_transformerblock(GPT_CONFIG_124M)

output = st(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)