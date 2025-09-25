import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from transformers import PretrainedConfig as _HFPretrainedConfig
except Exception:
    class _HFPretrainedConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def to_dict(self):
            return dict(self.__dict__)
        def get(self, key, default=None):
            return getattr(self, key, default)

class DummyConfig(_HFPretrainedConfig):
    """Config that mirrors your JSON and provides .get() for PEFT."""
    model_type = "dummy"

    def __init__(
        self,
        vocab_size,
        n_positions,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        activation_function="gelu",
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        attention_probs_dropout_prob=0.0,
        resid_dropout_prob=0.1,
        qkv_bias=True,
        grad_ckpt=False,
        tie_word_embeddings=True,
        torch_dtype="float32",
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.activation_function = activation_function
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.resid_dropout_prob = resid_dropout_prob
        self.qkv_bias = qkv_bias
        self.grad_ckpt = grad_ckpt

    def get(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_training_json(cls, cfg: dict):
        return cls(
            vocab_size=cfg["vocab_size"],
            n_positions=cfg["context_length"],
            hidden_size=cfg["emb_dim"],
            num_hidden_layers=cfg["n_layers"],
            num_attention_heads=cfg["n_heads"],
            activation_function=cfg.get("activation", "gelu"),
            layer_norm_eps=cfg.get("layer_norm_eps", 1e-5),
            initializer_range=cfg.get("initializer_range", 0.02),
            attention_probs_dropout_prob=cfg.get("attention_probs_dropout_prob", 0.0),
            resid_dropout_prob=cfg.get("drop_rate", 0.1),
            qkv_bias=cfg.get("qkv_bias", False),
            grad_ckpt=cfg.get("grad_ckpt", False),
            tie_word_embeddings=True,
        )

def build_activation(name: str):
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")

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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, qkv_bias, attn_drop, resid_drop, max_ctx):
        super().__init__()
        assert d_model % n_heads == 0, "hidden_size must be divisible by num_heads"
        self._attn_impl_logged = False

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)

        self._backend_cache = {}

        mask = torch.triu(torch.ones(max_ctx, max_ctx, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv.unbind(dim=2)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()

        use_fp16_attn = (
            q.dtype is torch.bfloat16
            and hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled")
            and torch.backends.cuda.mem_efficient_sdp_enabled()
        )

        dropout_p = self.attn_drop.p if self.training else 0.0

        if use_fp16_attn:
            y = F.scaled_dot_product_attention(
                q.to(torch.float16), k.to(torch.float16), v.to(torch.float16),
                attn_mask=None, is_causal=True, dropout_p=dropout_p
            ).to(torch.bfloat16)
            if not self._attn_impl_logged:
                print("[attn] SDPA (mem-efficient) via FP16 compute, model bf16")
                self._attn_impl_logged = True
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True, dropout_p=dropout_p
            )
            if not self._attn_impl_logged:
                dtype_name = str(q.dtype).split(".")[-1]
                print(f"[attn] SDPA auto-select for dtype={dtype_name}")
                self._attn_impl_logged = True

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_drop(y)
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
        a = self.ln1(x).contiguous()
        x = x + self.attn(a) * self.res_scale
        m = self.ln2(x).contiguous()
        x = x + self.mlp(m) * self.res_scale
        return x

class DummyModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.config = DummyConfig.from_training_json(cfg)

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

    def get_input_embeddings(self):
        return self.tok_emb

    def set_input_embeddings(self, new_emb):
        self.tok_emb = new_emb
        self.lm_head.weight = self.tok_emb.weight

    def prepare_inputs_for_generation(self, input_ids=None, attention_mask=None, **kwargs):
        out = {"input_ids": input_ids}
        if attention_mask is not None:
            out["attention_mask"] = attention_mask
        out.update(kwargs)
        return out

    def forward(self, idx=None, labels=None, input_ids=None, **kwargs):
        if idx is None:
            idx = input_ids
        assert idx is not None, "need idx or input_ids"
        assert idx.dtype == torch.long, "input ids must be torch.long"

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
            for b in self.blocks:
                x = checkpoint(lambda t, blk=b: blk(t), x, use_reentrant=False, preserve_rng_state=False)
        else:
            for b in self.blocks:
                x = b(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            labels = labels[:, :T]
            logits_flat = logits[:, :-1, :].reshape(-1, self.vocab_size)
            labels_flat = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

        return logits, loss

    def num_params(self, trainable_only=True):
        ps = (p for p in self.parameters() if (p.requires_grad or not trainable_only))
        return sum(p.numel() for p in ps)
