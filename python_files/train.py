import os, math, time, json, random, argparse, csv, warnings
from pathlib import Path
import datetime as dt

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import contextlib

from Dummy_Model import DummyModel
from npy_datasets import NpyTokensDataset
import logging

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    def sdpa_ctx_fast():
        return sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
except Exception:
    from torch.backends.cuda import sdp_kernel
    def sdpa_ctx_fast():
        return sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)


warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

SEED = 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {}

def _maybe_fuse_qkv_in_sd(sd: dict, model) -> dict:
    try:
        n_layers = len(getattr(model, "blocks", []))
    except Exception:
        return sd

    for i in range(n_layers):
        pre = f"blocks.{i}.attn."
        q_w = sd.get(pre + "q_proj.weight")
        k_w = sd.get(pre + "k_proj.weight")
        v_w = sd.get(pre + "v_proj.weight")
        if (pre + "qkv.weight") not in sd and (q_w is not None and k_w is not None and v_w is not None):
            sd[pre + "qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
            q_b = sd.get(pre + "q_proj.bias")
            k_b = sd.get(pre + "k_proj.bias")
            v_b = sd.get(pre + "v_proj.bias")
            if q_b is not None and k_b is not None and v_b is not None:
                sd[pre + "qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)
            for suf in ["q_proj", "k_proj", "v_proj"]:
                sd.pop(pre + f"{suf}.weight", None)
                sd.pop(pre + f"{suf}.bias", None)
    return sd



def _unwrap_model_for_state_dict(model: nn.Module) -> nn.Module:
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    if hasattr(model, "module"):
        return model.module
    return model

def _strip_prefix_in_state_dict(sd: dict, prefix: str) -> dict:
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


def worker_init_fn(worker_id: int):
    seed = (torch.initial_seed() + worker_id) % (2**31 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)

def param_groups_weight_decay(module: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = n.endswith("bias")
        is_norm = ("norm" in n.lower()) or ("ln" in n.lower())
        (no_decay if (is_bias or is_norm or p.ndim <= 1) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]

def build_cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = min(max(t, 0.0), 1.0)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class EMAMeter:
    def __init__(self, beta=0.9): self.beta, self.val, self.inited = beta, 0.0, False
    def update(self, x):
        x = float(x)
        if not self.inited: self.val, self.inited = x, True
        else: self.val = self.beta*self.val + (1-self.beta)*x
        return self.val

class TopK:
    def __init__(self, k=5):
        self.k = k
        self.heap = []
    def add(self, loss, update, path):
        import heapq
        item = (-float(loss), int(update), str(path))
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        else:
            if item > self.heap[0]:
                heapq.heapreplace(self.heap, item)
    def best(self):
        return sorted([(-l, u, p) for (l, u, p) in self.heap], key=lambda x: x[0])

def _to_byte_tensor(x):
    if x is None: return None
    if isinstance(x, torch.ByteTensor): return x.cpu()
    if isinstance(x, torch.Tensor):     return x.detach().to(dtype=torch.uint8, device="cpu")
    if isinstance(x, (bytes, bytearray)): return torch.tensor(list(x), dtype=torch.uint8)
    if isinstance(x, (list, tuple, np.ndarray)): return torch.tensor(x, dtype=torch.uint8)
    return None

def make_run_dirs(base):
    base = Path(base)
    ckpt = base / "checkpoints"
    plots = base / "plots"
    logs = base / "logs"
    for d in (ckpt, plots, logs): d.mkdir(parents=True, exist_ok=True)
    return base, ckpt, plots, logs

def manifest_load_or_init(pretrain_dir: Path):
    path = pretrain_dir / "manifest.json"
    if not path.exists():
        manifest = {
            "created_utc": dt.datetime.now(dt.UTC).isoformat() + "Z",
            "chunks_completed": 0,
            "last_update": 0,
            "best_val": None,
            "cfg": cfg,
        }
        path.write_text(json.dumps(manifest, indent=2))
    else:
        manifest = json.loads(path.read_text())
    return manifest, path

def get_manifest_best(manifest, fallback):
    x = manifest.get("best_val")
    return (float(x) if x is not None else (float(fallback) if fallback is not None else float("inf")))

def history_init(csv_path: Path):
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["update","split","loss","ppl","lr","tps","gnorm","scale","tokens_seen","utc"]
            )

def log_train_row(csv_path, update, loss, lr, tps, gnorm, scale, tokens_seen):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([
            int(update),"train",float(loss),"",float(lr),float(tps),
            float(gnorm),float(scale),int(tokens_seen),dt.datetime.now(dt.UTC).isoformat()+"Z"
        ])

def log_val_row(csv_path, update, loss, ppl, tokens_seen):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([
            int(update),"val",float(loss),float(ppl),"","","","",
            int(tokens_seen),dt.datetime.now(dt.UTC).isoformat()+"Z"
        ])

def save_plot(history_csv: Path, plots_dir: Path, eff_tokens_per_update: int, save_tag="latest"):
    try:
        
        if not history_csv.exists(): return
        df = pd.read_csv(history_csv)
        if df.empty: return

        df["tokens_seen"] = df["update"] * eff_tokens_per_update
        tr = df[df["split"]=="train"].copy()
        vl = df[df["split"]=="val"].copy()

        fig, ax = plt.subplots(figsize=(7,3.2), dpi=120)
        ax.plot(tr["update"], tr["loss"], label="Training loss")
        if not vl.empty:
            ax.plot(vl["update"], vl["loss"], linestyle="--", label="Validation loss")
        ax.set_xlabel("Updates"); ax.set_ylabel("Loss"); ax.legend(loc="best")

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        xticks = ax.get_xticks()
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f"{int(x*eff_tokens_per_update/1000):,}k" for x in xticks])
        ax2.set_xlabel("Tokens seen")

        out = plots_dir / f"loss_curve_{save_tag}.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        print(f"🖼️ saved plot → {out}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")

def save_checkpoint(ckpt_dir: Path, manifest: dict, manifest_path: Path,
                    model, optimizer, scheduler, scaler, update, val_loss=None, tag="latest", light=False):
    
    base_model = _unwrap_model_for_state_dict(model)

    if tag == "latest":
        path = ckpt_dir / "latest.pt"
    elif tag == "best":
        path = ckpt_dir / "best.pt"
    else:
        path = ckpt_dir / f"{tag}.pt"

    payload = {
        "update": int(update),
        "val_loss": (None if val_loss is None else float(val_loss)),
        "cfg": cfg,
        "timestamp": time.time(),
        "model": base_model.state_dict(),
    }
    if not light:
        payload.update({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict()
                if (scaler is not None and getattr(scaler, "is_enabled", lambda: False)()) else None,
            "rng": {
                "torch_cpu": torch.random.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        })
    torch.save(payload, path)

    manifest["last_update"] = int(update)
    if val_loss is not None:
        m_best = manifest.get("best_val")
        if (m_best is None) or (val_loss < m_best):
            manifest["best_val"] = float(val_loss)
    manifest["cfg"] = cfg
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return str(path)

def load_state_dict_safely(model, raw_sd):
    sd = raw_sd.get("model", raw_sd)

    pe_key = "pos_emb.weight"
    if pe_key in sd:
        old_len = sd[pe_key].shape[0]
        new_len = model.pos_emb.num_embeddings
        if old_len != new_len:
            if old_len > new_len:
                sd[pe_key] = sd[pe_key][:new_len, :]
            else:
                pad = torch.zeros(new_len - old_len, sd[pe_key].shape[1],
                                  dtype=sd[pe_key].dtype, device=sd[pe_key].device)
                sd[pe_key] = torch.cat([sd[pe_key], pad], dim=0)
    
    sd = _maybe_fuse_qkv_in_sd(sd, model)

    base = _unwrap_model_for_state_dict(model)
    missing, unexpected = base.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict load: missing={len(missing)} unexpected={len(unexpected)}")
        for k in list(missing)[:5]:    print("  missing:", k)
        for k in list(unexpected)[:5]: print("  unexpected:", k)

    return missing, unexpected


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, map_location):
    if not path.exists():
        print("no checkpoint found, starting fresh")
        return 1, None, False

    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    raw_sd = ckpt.get("model", ckpt)
    raw_sd = _strip_prefix_in_state_dict(raw_sd, "_orig_mod.")
    raw_sd = _strip_prefix_in_state_dict(raw_sd, "module.")
    missing, unexpected = load_state_dict_safely(model, raw_sd)

    opt_sd = ckpt.get("optimizer")
    if opt_sd is not None:
        try:
            optimizer.load_state_dict(opt_sd)
        except Exception as e:
            print(f"[warn] optimizer state incompatible; using fresh optimizer. reason: {e}")
    else:
        print("[warn] optimizer state not in checkpoint (light resume).")

    had_sched = False
    sch_sd = ckpt.get("scheduler")
    if sch_sd is not None:
        try:
            scheduler.load_state_dict(sch_sd)
            had_sched = True
        except Exception as e:
            print(f"[warn] scheduler restore failed: {e}")
    else:
        print("[warn] scheduler state not in checkpoint (light resume).")

    sc_sd = ckpt.get("scaler")
    if scaler is not None and sc_sd is not None:
        try:
            scaler.load_state_dict(sc_sd)
        except Exception as e:
            print(f"[warn] AMP scaler restore failed: {e}")
    elif scaler is not None and sc_sd is None:
        print("[warn] AMP scaler state not in checkpoint (light resume).")

    rng = ckpt.get("rng")
    if rng is not None:
        try:
            cpu_state = _to_byte_tensor(rng.get("torch_cpu"))
            if cpu_state is not None:
                torch.random.set_rng_state(cpu_state)
        except Exception as e:
            print(f"[warn] CPU RNG restore skipped: {e}")
        try:
            cuda_states = rng.get("torch_cuda")
            if torch.cuda.is_available() and cuda_states is not None:
                if isinstance(cuda_states, (list, tuple)):
                    cuda_states = [_to_byte_tensor(s) for s in cuda_states]
                    torch.cuda.set_rng_state_all(cuda_states)
                else:
                    torch.cuda.set_rng_state(_to_byte_tensor(cuda_states))
        except Exception as e:
            print(f"[warn] CUDA RNG restore skipped: {e}")
    else:
        print("[warn] RNG state not in checkpoint (light resume).")

    start = int(ckpt.get("update", 0)) + 1
    best  = ckpt.get("val_loss", None)

    if not had_sched:
        try:
            scheduler.last_epoch = start - 1
        except Exception as e:
            print(f"[warn] couldn't fast-forward scheduler: {e}")

    print(f"🔄 resumed from {path} @ update {start-1} (best_val={best})")
    return start, best, had_sched

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None, max_batches=5):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    it = iter(loader)
    for _ in range(max_batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            break
        xb = xb.to(device, non_blocking=True).long()
        yb = yb.to(device, non_blocking=True).long()
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if (amp_dtype is not None and device.type == "cuda") else contextlib.nullcontext()
        )

        with autocast_ctx:
            _, loss = model(input_ids=xb, labels=yb)
        B, T = yb.shape
        total_loss += loss.item() * (B * T)
        total_tokens += (B * T)
    model.train()
    avg = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg))
    return avg, ppl

def _cfg_diff(a, b, ignore=("grad_ckpt", "initializer_range")):
    ks = set(a) | set(b)
    diffs = {}
    for k in ks:
        if k in ignore:
            continue
        va, vb = a.get(k), b.get(k)
        if va != vb:
            diffs[k] = {"was": va, "now": vb}
    return diffs

def _cuda_ev():
    return (torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))

def profile_one_train_update(model, optimizer, scaler, train_loader, amp_dtype, device):
    """Profiles forward/backward for a single micro-batch WITHOUT updating weights."""
    if device.type != "cuda":
        print("[profile] CUDA not available; skipping")
        return

    model.train()
    it = iter(train_loader)
    xb, yb = next(it)
    xb = xb.to(device, non_blocking=True).long()
    yb = yb.to(device, non_blocking=True).long()

    f_start, f_end = _cuda_ev()
    b_start, b_end = _cuda_ev()

    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)

    ctx = (torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
           if (amp_dtype is not None and device.type == "cuda") else contextlib.nullcontext())
    f_start.record()
    with ctx:
        _, loss = model(input_ids=xb, labels=yb)
    f_end.record()

    b_start.record()
    if scaler and getattr(scaler, "is_enabled", lambda: False)():
        scaler.scale(loss).backward()
    else:
        loss.backward()
    b_end.record()

    torch.cuda.synchronize()
    f = f_start.elapsed_time(f_end)
    b = b_start.elapsed_time(b_end)
    print(f"[profile] forward={f:.1f}ms  backward={b:.1f}ms  (no optimizer.step)")

def main():
    if os.environ.get("TORCH_LOGS", "") == "error":
        os.environ.pop("TORCH_LOGS", None)

    ap = argparse.ArgumentParser()

    ap.add_argument("--clip-every", type=int, default=1)

    
    ap.add_argument("--train-blocks", type=str, default="")
    ap.add_argument("--valid-blocks", type=str, default="")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--config", type=str, default="")
    
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--persistent", action="store_true")
    ap.add_argument("--micro-bsz", type=int, default=8)
    
    ap.add_argument("--total-updates", type=int, default=24_000)
    ap.add_argument("--pilot-frac", type=float, default=0.05)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--periodic-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    
    ap.add_argument("--amp", choices=["fp16","bf16","none"], default="fp16")
    ap.add_argument("--torch-compile", action="store_true")
    ap.add_argument("--compile-mode", type=str, default="reduce-overhead")
    ap.add_argument("--resume-from", type=str, default="latest")
    args = ap.parse_args()

    global cfg
    with open(args.config, "r") as f:
        cfg.clear()
        cfg.update(json.load(f))

    print("Loaded config:", json.dumps(cfg, indent=2))

    print("device:", device)

    assert Path(args.train_blocks).exists() and Path(args.valid_blocks).exists()
    train_ds = NpyTokensDataset(args.train_blocks, cfg["context_length"])
    valid_ds = NpyTokensDataset(args.valid_blocks, cfg["context_length"])

    base_kwargs = dict(
        batch_size=args.micro_bsz,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(args.workers > 0 and args.persistent),
        shuffle=True,
    )
    valid_kwargs = dict(base_kwargs)
    valid_kwargs["shuffle"] = False
    if args.workers > 0:
        base_kwargs["prefetch_factor"] = args.prefetch
        base_kwargs["worker_init_fn"] = worker_init_fn
        valid_kwargs["prefetch_factor"] = args.prefetch
        valid_kwargs["worker_init_fn"] = worker_init_fn
        
    try:
        base_kwargs["pin_memory_device"] = "cuda"
        valid_kwargs["pin_memory_device"] = "cuda"
    except TypeError:
        pass

    train_loader = DataLoader(train_ds, **base_kwargs)
    valid_loader = DataLoader(valid_ds, **valid_kwargs)

    model = DummyModel(cfg).to(device)
    print("params:", sum(p.numel() for p in model.parameters()))

    if args.torch_compile:
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode=args.compile_mode)
            print("torch.compile enabled.")
        except Exception as e:
            print("compile skipped:", e)

    BASE_LR, WEIGHT_DECAY = 5e-4, 0.10
    opt_groups = param_groups_weight_decay(model, WEIGHT_DECAY)
    try:
        optimizer = torch.optim.AdamW(opt_groups, lr=BASE_LR, betas=(0.9,0.95), eps=1e-8, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(opt_groups, lr=BASE_LR, betas=(0.9,0.95), eps=1e-8)

    WARMUP_STEPS = max(1, int(0.10 * args.total_updates))
    scheduler = build_cosine_with_warmup(
        optimizer,
        warmup_steps=WARMUP_STEPS,
        total_steps=args.total_updates,
        min_lr_ratio=args.min_lr_ratio
    )
    print(f"[lr] BASE_LR=5e-4  warmup={WARMUP_STEPS}  min_lr_ratio={args.min_lr_ratio}")

    amp_dtype = None
    scaler = None
    use_cuda = (device.type == "cuda")
    if args.amp == "fp16" and use_cuda:
        amp_dtype = torch.float16
        scaler = GradScaler(enabled=True)
    elif args.amp == "bf16" and use_cuda:
        amp_dtype = torch.bfloat16
        scaler = GradScaler(enabled=False)
    else:
        amp_dtype = None
        scaler = GradScaler(enabled=False)


    xb, yb = next(iter(train_loader))
    assert xb.dtype == torch.int64 and yb.dtype == torch.int64
    assert xb.shape[1] == cfg["context_length"] and yb.shape == xb.shape
    print("batch OK:", xb.shape, yb.shape)

    PRETRAIN_DIR, CKPT_DIR, PLOTS_DIR, LOGS_DIR = make_run_dirs(args.run_dir)
    manifest, MANIFEST_PATH = manifest_load_or_init(PRETRAIN_DIR)
    prev_total = int(manifest.get("schedule", {}).get("total_updates", 0))
    if args.total_updates > prev_total:
        manifest.setdefault("schedule", {})
        manifest["schedule"]["total_updates"] = args.total_updates
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    resuming = any((CKPT_DIR / name).exists() for name in ("latest.pt", "best.pt"))

    manifest.setdefault("schedule", {})
    if resuming:
        diff = _cfg_diff(manifest.get("cfg", {}), cfg)
        if diff:
            raise RuntimeError("Config changed since checkpoint:\n" + json.dumps(diff, indent=2))
    
    HISTORY_CSV = LOGS_DIR / "history.csv"
    history_init(HISTORY_CSV)

    rp = Path(args.resume_from)
    if rp.exists():
        resume_path = rp
    elif args.resume_from in ("latest", "best"):
        resume_path = CKPT_DIR / f"{args.resume_from}.pt"
    else:
        name = rp.name
        if not name.endswith(".pt"):
            name += ".pt"
        resume_path = CKPT_DIR / name

    if not resume_path.exists():
        print(f"[warn] {resume_path} not found; falling back to latest.pt")
        resume_path = CKPT_DIR / "latest.pt"

    start_update, best_val, had_sched = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, device)
    cur_lr = optimizer.param_groups[0]['lr']
    print(f"[resume] lr now = {cur_lr:.2e} at upd {start_update-1}")

    val0, ppl0 = evaluate(model, valid_loader, device, amp_dtype, max_batches=20)
    print(f"[resume] baseline val_loss={val0:.4f}  ppl={ppl0:.2f}  at upd {start_update-1}")

    if start_update > 1 and device.type == "cuda":
        profile_one_train_update(model, optimizer, scaler, train_loader, amp_dtype, device)

    if best_val is None: best_val = float("inf")

    manifest.setdefault("schedule", {})
    prev_total = int(manifest["schedule"].get("total_updates", 0))
    if args.total_updates > prev_total:
        manifest["schedule"]["total_updates"] = args.total_updates
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    if (start_update > 1) and (not had_sched):
        sched_cfg = manifest.get("schedule", {})
        target_total = max(int(sched_cfg.get("total_updates", args.total_updates)), int(args.total_updates))
        current_step = start_update - 1
        if target_total > current_step:
            remaining = target_total - current_step
            cur_lr = optimizer.param_groups[0]["lr"]
            for pg in optimizer.param_groups:
                pg["initial_lr"] = cur_lr
                pg["lr"] = cur_lr
            scheduler = build_cosine_with_warmup(
                optimizer, warmup_steps=0, total_steps=remaining, min_lr_ratio=args.min_lr_ratio
            )

    if start_update > 1:
        print(f"🔄 RESUMING TRAINING from update {start_update-1} "
            f"(latest.best_val={best_val}, manifest.best_val={manifest.get('best_val')}) in {PRETRAIN_DIR}")
    else:
        print("🚀 Starting fresh training run.")

    topk = TopK(k=5)
    def save_periodic(update, val_loss, every=args.periodic_every):
        if every > 0 and update % every == 0:
            tag = f"u{update:07d}"
            save_checkpoint(CKPT_DIR, manifest, MANIFEST_PATH, model, optimizer, scheduler, scaler,
                update, val_loss, tag=tag, light=True)
            print(f"💾 saved periodic checkpoint: {tag}")

    def register_val_result(update, val_loss):
        save_checkpoint(CKPT_DIR, manifest, MANIFEST_PATH, model, optimizer, scheduler, scaler, update, val_loss, "latest")
        current_best = getattr(register_val_result, "_best", float("inf"))
        if val_loss < current_best:
            save_checkpoint(CKPT_DIR, manifest, MANIFEST_PATH, model, optimizer, scheduler, scaler, update, val_loss, "best")
            register_val_result._best = val_loss
            print(f"⭐ new best: {val_loss:.4f} @ upd {update}")
        tag = f"u{update:07d}"
        topk.add(val_loss, update, str(CKPT_DIR / f"{tag}.pt"))

    register_val_result._best = get_manifest_best(manifest, best_val)

    TOTAL_UPDATES   = args.total_updates
    PILOT_FRAC      = args.pilot_frac
    CHUNKS          = args.chunks
    EVAL_EVERY      = args.eval_every
    LOG_EVERY       = args.log_every
    ACCUM           = args.accum
    CTX             = cfg["context_length"]
    MICRO_BSZ       = train_loader.batch_size
    EFFECTIVE_TOKENS_PER_UPDATE = CTX * MICRO_BSZ * ACCUM

    print(f"Plan → TOTAL_UPDATES={TOTAL_UPDATES} (pilot={int(PILOT_FRAC*TOTAL_UPDATES)}, "
          f"{CHUNKS}×chunks). eff tokens/update ≈ {EFFECTIVE_TOKENS_PER_UPDATE:,}.")

    BASELINE_WINDOW       = 50
    TOKENS_SEC_ALERT_DROP = 0.30
    LOSS_SPIKE_X          = 2.0
    CONSEC_FOR_ALERT      = 8

    from tqdm.auto import tqdm
    postfix = {}

    def train_for_updates(num_updates, tag):
        nonlocal start_update, best_val
        end_update = start_update + num_updates - 1

        tps_ema  = EMAMeter(0.8)
        loss_ema = EMAMeter(0.9)
        base_tps = None
        base_loss= None
        tps_bad  = 0
        loss_bad = 0

        model.train()
        loader_iter = iter(train_loader)
        pbar = tqdm(range(start_update, end_update + 1), desc=f"Training [{tag}]", unit="upd")

        for update in pbar:
            optimizer.zero_grad(set_to_none=True)
            upd_t0 = time.time()
            loss_accum = 0.0

            for _ in range(ACCUM):
                dl_t0 = time.time()
                try:
                    xb, yb = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    xb, yb = next(loader_iter)
                
                dl_ms = (time.time() - dl_t0) * 1000

                xb = xb.to(device, non_blocking=True).long()
                yb = yb.to(device, non_blocking=True).long()

                if amp_dtype is not None and device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                       _, loss = model(input_ids=xb, labels=yb)
                else:
                    _, loss = model(input_ids=xb, labels=yb)


                loss = loss / ACCUM
                if scaler and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accum += float(loss)

            try:
                if scaler and scaler.is_enabled():
                    scaler.unscale_(optimizer)
            except RuntimeError:
                pass

            if args.clip_every > 0 and (update % args.clip_every == 0):
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            else:
                gnorm = torch.tensor(0.0)

            try:
                if scaler and scaler.is_enabled():
                    scaler.step(optimizer)
                else:
                    optimizer.step()
            except AssertionError:
                if scaler and scaler.is_enabled(): scaler.update()
                scheduler.step()
                continue

            if scaler and scaler.is_enabled():
                scaler.update()
            scheduler.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            dt_upd  = max(time.time() - upd_t0, 1e-6)
            tps     = EFFECTIVE_TOKENS_PER_UPDATE / dt_upd
            tps_sm  = tps_ema.update(tps)
            loss_sm = loss_ema.update(loss_accum)
            gpu_gb  = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
            amp_sc  = float(getattr(scaler, "get_scale", lambda: 1.0)())

            postfix.update({
                "loss": f"{loss_sm:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "tps": f"{tps_sm:,.0f}",
                "gpu": f"{gpu_gb:.2f}GB",
                "gnorm": f"{float(gnorm):.2f}",
                "scale": f"{amp_sc:.0f}",
                "dl": f"{dl_ms:.0f}ms",
            })

            tokens_seen = update * EFFECTIVE_TOKENS_PER_UPDATE
            log_train_row(HISTORY_CSV, update, loss_sm, optimizer.param_groups[0]['lr'], tps_sm,
                          float(gnorm), float(amp_sc), tokens_seen)

            seen = update - start_update + 1
            if seen == BASELINE_WINDOW:
                base_tps  = tps_sm
                base_loss = loss_sm

            if base_tps is not None and base_tps > 0:
                tps_bad = tps_bad + 1 if tps_sm < (1.0 - TOKENS_SEC_ALERT_DROP) * base_tps else 0
            if base_loss is not None and base_loss > 0:
                loss_bad = loss_bad + 1 if loss_sm > LOSS_SPIKE_X * base_loss else 0

            if (tps_bad >= CONSEC_FOR_ALERT) or (loss_bad >= CONSEC_FOR_ALERT):
                print(f"\n⚠️  ALERT @ upd {update}: "
                      f"{'tps drop' if tps_bad>=CONSEC_FOR_ALERT else ''}"
                      f"{' and ' if tps_bad>=CONSEC_FOR_ALERT and loss_bad>=CONSEC_FOR_ALERT else ''}"
                      f"{'loss spike' if loss_bad>=CONSEC_FOR_ALERT else ''}. "
                      f"tps≈{tps_sm:,.0f} (base≈{base_tps:,.0f}), loss≈{loss_sm:.4f} (base≈{base_loss:.4f})")
                val_loss, val_ppl = evaluate(model, valid_loader, device, amp_dtype, max_batches=5)
                print(f"   immediate eval: val_loss={val_loss:.4f}  ppl={val_ppl:.2f}")
                save_checkpoint(
                    CKPT_DIR, manifest, MANIFEST_PATH, model, optimizer, scheduler, scaler,
                    update, val_loss, tag=f"alert_u{update:07d}", light=True
                )
                save_plot(HISTORY_CSV, PLOTS_DIR, EFFECTIVE_TOKENS_PER_UPDATE, save_tag=f"alert_u{update:07d}")
                tps_bad = loss_bad = 0

            if update % LOG_EVERY == 0:
                pbar.set_postfix(**postfix)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)

            if EVAL_EVERY > 0 and update % EVAL_EVERY == 0:
                val_loss, val_ppl = evaluate(model, valid_loader, device, amp_dtype, max_batches=5)
                print(f"\n[upd {update}] val_loss={val_loss:.4f}  ppl={val_ppl:.2f}")
                log_val_row(HISTORY_CSV, update, val_loss, val_ppl, tokens_seen)
                if args.periodic_every > 0 and update % args.periodic_every == 0:
                    save_periodic(update, val_loss, every=args.periodic_every)
                register_val_result(update, val_loss)
                save_plot(HISTORY_CSV, PLOTS_DIR, EFFECTIVE_TOKENS_PER_UPDATE, save_tag=f"u{update:07d}")

            start_update = update + 1

        save_checkpoint(
            CKPT_DIR, manifest, MANIFEST_PATH, model, optimizer, scheduler, scaler,
            start_update - 1, None, tag=f"u{start_update-1:07d}", light=True
        )

        if tag.startswith("chunk"):
            manifest["chunks_completed"] = int(manifest.get("chunks_completed", 0)) + 1
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
        print(f"✅ Finished {tag}. Reached update {start_update-1}.")

    pilot_updates = max(1, int(PILOT_FRAC * TOTAL_UPDATES))
    chunk_updates = max(1, (TOTAL_UPDATES - pilot_updates) // CHUNKS)

    if start_update <= pilot_updates:
        todo = pilot_updates - (start_update - 1)
        train_for_updates(todo, tag="pilot")
    else:
        print("Pilot already completed (resume detected).")

    for i in range(1, CHUNKS + 1):
        target_end = pilot_updates + i * chunk_updates
        if start_update <= target_end:
            todo = target_end - (start_update - 1)
            train_for_updates(todo, tag=f"chunk{i}")
        else:
            print(f"chunk{i} already completed (resume detected).")

    remaining = TOTAL_UPDATES - (start_update - 1)
    if remaining > 0:
        train_for_updates(remaining, tag="tail")

    print("🏁 Full run plan completed (within this step budget).")

if __name__ == "__main__":    
    mp.set_start_method("spawn", force=True)
    try:
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    
    main()