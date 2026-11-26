import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchedMultiheadAttention(nn.Module):
    """nn.MultiheadAttention with separate q/k/v/out projections, so LoRA can adapt them."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    @torch.no_grad()
    def load_from_pytorch_mha(self, mha: nn.MultiheadAttention):
        """Copy weights from an existing nn.MultiheadAttention."""
        d = self.embed_dim
        w = mha.in_proj_weight
        b = mha.in_proj_bias
        self.q_proj.weight.copy_(w[0:d])
        self.k_proj.weight.copy_(w[d:2*d])
        self.v_proj.weight.copy_(w[2*d:3*d])
        if b is not None:
            self.q_proj.bias.copy_(b[0:d])
            self.k_proj.bias.copy_(b[d:2*d])
            self.v_proj.bias.copy_(b[2*d:3*d])
        else:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
        self.out_proj.weight.copy_(mha.out_proj.weight)
        if mha.out_proj.bias is not None:
            self.out_proj.bias.copy_(mha.out_proj.bias)
        else:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        L, N, E = query.shape
        S = key.shape[0]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.reshape(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.reshape(S, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.reshape(S, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.permute(2, 0, 1, 3).reshape(L, N, E)

        out = self.out_proj(attn_out)
        return out, None   # match nn.MultiheadAttention return signature


def patch_attention_in_transformer(transformer):
    """Replace every nn.MultiheadAttention in a CLIP transformer with PatchedMultiheadAttention."""
    for block in transformer.resblocks:
        old_attn = block.attn
        d_model = old_attn.embed_dim
        n_head = old_attn.num_heads
        ref_param = next(old_attn.parameters())

        new_attn = PatchedMultiheadAttention(d_model, n_head)
        new_attn.load_from_pytorch_mha(old_attn)
        new_attn = new_attn.to(device=ref_param.device, dtype=ref_param.dtype)
        block.attn = new_attn
    return transformer