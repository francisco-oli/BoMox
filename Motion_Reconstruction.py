import sys
import av
import os
import json
import numpy as np
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas
from tqdm.auto import tqdm
from torch.cuda.amp import autocast

class Attention_Head(nn.Module):
  def __init__(self, emb_dim, head_size):
    super().__init__()

    self.head_size = head_size
    self.emb_dim = emb_dim

    self.key = nn.Linear(emb_dim, head_size, bias=False)
    self.query = nn.Linear(emb_dim, head_size, bias=False)
    self.value = nn.Linear(emb_dim, head_size, bias=False)

  def forward(self, x):

    q = self.query(x)
    k = self.key(x)
    v = self.value(x)

    out = F.scaled_dot_product_attention(q, k, v)

    return out

class MultiHead_Attention (nn.Module):
  def __init__(self, num_heads, emb_dim):
    super().__init__()

    self.head_size = emb_dim // num_heads
    self.heads = nn.ModuleList([Attention_Head(emb_dim, self.head_size) for _ in range(num_heads)])

  def forward(self, x):

    return torch.cat([h(x) for h in self.heads], dim=-1)

class Transfomer_Block(nn.Module):
  def __init__(self, num_heads, emb_dim):
    super().__init__()

    self.attention = MultiHead_Attention(num_heads, emb_dim)
    self.ln = nn.LayerNorm(emb_dim)
    self.ffn = nn.Sequential(
        nn.Linear(emb_dim, emb_dim * 4),
        nn.GELU(),
        nn.Linear(emb_dim * 4, emb_dim)
    )
    self.ln2 = nn.LayerNorm(emb_dim)

  def forward(self, x):

    x = x + self.attention(x)

    x = self.ln(x)
    x = x + self.ffn(x)
    x = self.ln2(x)

    return x
  
class Reconstruction_Transformer(nn.Module):
  def __init__(self, num_layers = 3, num_attention_heads = 4, window_size = 512, num_global_tokens = 8):
        super().__init__()

        self.window_size = window_size #Length of replicated motion sequence
        self.embedding_dim = 512 #Dim of each sequence's embedding
        self.num_global_tokens = num_global_tokens

        self.projection = nn.Sequential(
            nn.LayerNorm(178 * 1024),
            nn.Linear(178 * 1024, self.embedding_dim),
            nn.GELU()
        )

        # ===== Global context tokens, replacing FiLM =====
        # final_emb (B, embedding_dim) is expanded into `num_global_tokens`
        # distinct vectors -- multiple "facets" of the global representation
        # the decoder can attend into individually, rather than a single
        # flat scale/shift applied uniformly to every position. Attention
        # over a single key/value trivially reduces to always using that
        # one value (softmax over 1 item = weight 1.0), so a genuine
        # multi-token expansion is what makes this real attention rather
        # than a disguised linear layer.
        self.global_token_proj = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, num_global_tokens * self.embedding_dim)
        )
        # Small learned embedding added to every global token so the
        # attention layers can distinguish "this is global context" from
        # "this is a real timestep" -- analogous to segment/type embeddings
        # in BERT-style architectures. Global tokens carry no notion of
        # time, so they intentionally do NOT receive positional_encoding.
        self.global_token_type_embedding = nn.Parameter(
            torch.randn(1, num_global_tokens, self.embedding_dim) * 0.02
        )
        # ===== END global context tokens =====

        self.positional_encoding = nn.Parameter(torch.randn(1, window_size, self.embedding_dim) * 0.02)

        #Self Attention
        self.layers = nn.ModuleList(Transfomer_Block(num_attention_heads, self.embedding_dim) for _ in range(num_layers))

        #MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 178)
        )

  def forward(self, batch, final_emb, unpooled_seq):
    smpl_batch, masks = batch

    # 1. Format the Unpooled MOMENT Timeline
    B = unpooled_seq.size(0)
    seq = unpooled_seq.permute(0, 2, 1, 3)
    seq = seq.reshape(B, 64, 178 * 1024)
    seq = self.projection(seq)

    # Expand the 64 patches back into 512 individual frames
    seq = torch.repeat_interleave(seq, repeats=8, dim=1)  # (B, 512, embedding_dim)

    # 2. Build the global context tokens from final_emb
    global_tokens = self.global_token_proj(final_emb)  # (B, num_global_tokens * embedding_dim)
    global_tokens = global_tokens.view(B, self.num_global_tokens, self.embedding_dim)
    global_tokens = global_tokens + self.global_token_type_embedding

    # 3. Positional encoding on the real timesteps only
    seq = seq + self.positional_encoding

    # 4. Concatenate: patch-token sequence now has 512 + num_global_tokens
    # positions. Self-attention over this concatenated sequence lets every
    # timestep attend into the global tokens with content- and
    # position-specific weights, and lets the global tokens attend back
    # into the timesteps -- genuine bidirectional attention, not a fixed
    # modulation.
    x = torch.cat([seq, global_tokens], dim=1)  # (B, 512 + num_global_tokens, embedding_dim)

    for layer in self.layers:
      x = layer(x)

    # 5. Drop the global tokens before the output head -- only the 512
    # real timestep positions get decoded into pose predictions.
    x = x[:, :self.window_size, :]

    output = self.mlp(x)

    return output
