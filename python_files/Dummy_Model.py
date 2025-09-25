import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    SDPA_CTX = sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    print("using sdpa_kernel")
except Exception:
    from torch.backends.cuda import sdp_kernel
    SDPA_CTX = sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)
    print("fallback sdp_kernel")



def build_activation(name: str):
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")

# commented out one are the original implementations
# i've discovered missmatch between original 10000 one and new +14000 one
# so, i refactored it to use a single linear layer for qkv projection
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, qkv_bias, attn_drop, resid_drop, max_ctx):
        super().__init__()
        assert d_model % n_heads == 0, "hidden_size must be divisible by num_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)

        mask = torch.triu(torch.ones(max_ctx, max_ctx, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):

        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv.unbind(dim=2)
        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        dropout_p = self.attn_drop.p if self.training else 0.0

        try:
            with SDPA_CTX:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    is_causal=True,
                    dropout_p=dropout_p,
                )
        except Exception:
            d = self.head_dim
            att = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(d)
            cm = self.causal_mask[:T, :T].to(att.device)
            att = att.masked_fill(cm, float("-inf"))
            att = torch.softmax(att, dim=-1)
            if self.training and self.attn_drop.p > 0:
                att = F.dropout(att, p=self.attn_drop.p)
            y = (att @ v.float()).to(q.dtype)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out_proj(y))
        return y
    
class MLP(nn.Module):
    def __init__(self, d_model, d_ff, drop, activation="gelu"):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff)
        self.act = build_activation(activation)
        self.proj = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg["emb_dim"]
        n_heads = cfg["n_heads"]
        d_ff = cfg.get("intermediate_size", 4 * d_model)
        drop = cfg.get("drop_rate", 0.1)
        attn_drop = cfg.get("attention_probs_dropout_prob", drop)
        qkv_bias = cfg.get("qkv_bias", False)
        max_ctx = cfg["context_length"]
        activation = cfg.get("activation", "gelu")
        eps = cfg.get("layer_norm_eps", 1e-5)

        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = MultiHeadAttention(d_model, n_heads, qkv_bias, attn_drop, drop, max_ctx)
        self.ln2 = nn.LayerNorm(d_model, eps=eps)
        self.mlp = MLP(d_model, d_ff, drop, activation)
        self.res_scale = 1.0 / math.sqrt(2 * cfg["n_layers"])

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) * self.res_scale
        x = x + self.mlp(self.ln2(x)) * self.res_scale
        return x

class DummyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg["vocab_size"]
        self.context_length = cfg["context_length"]
        d_model = cfg["emb_dim"]
        n_layers = cfg["n_layers"]
        drop = cfg.get("drop_rate", 0.1)
        eps = cfg.get("layer_norm_eps", 1e-5)
        init_std = cfg.get("initializer_range", 0.02)

        self.tok_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.context_length, d_model)
        self.drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model, eps=eps)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.grad_ckpt = bool(cfg.get("grad_ckpt", False))
        self.apply(lambda m: _init_weights(m, init_std))

    def forward(self, idx=None, input_ids=None, labels=None, **_):
        if idx is None:
            idx = input_ids
        assert idx is not None, "either idx or input_ids must be provided"
        B, T = idx.shape
        if T > self.context_length:
            idx = idx[:, -self.context_length:]
            T = idx.shape[1]

        T = min(T, self.context_length)
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx[:, :T]) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        use_ckpt = self.grad_ckpt and self.training
        if use_ckpt:
            # rely on outer autocast if it’s on
            for b in self.blocks:
                x = checkpoint(lambda t, blk=b: blk(t),
                            x, use_reentrant=False, preserve_rng_state=False)
        else:
            for b in self.blocks:
                x = b(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            labels = labels[:, :T]
            assert labels.dtype == torch.long
            logits_flat = logits[:, :-1, :].reshape(-1, self.vocab_size)
            labels_flat = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

        return logits, loss
    
    def num_params(self, trainable_only=True):
        ps = (p for p in self.parameters() if (p.requires_grad or not trainable_only))
        return sum(p.numel() for p in ps)

def _init_weights(module, std):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)