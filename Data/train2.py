# -*- coding: utf-8 -*-
"""
FlowMo latent denoiser — STEP-BASED trainer (residual training, curriculum, AMP, EMA)
- Trains for a fixed number of optimizer steps (total_steps), not epochs.
- Cycles data from small shards indefinitely to reach enough updates.
- Residual (noise) prediction ε = x_noisy - x_clean; reconstruct x̂ = x_noisy - ε̂.
- Accurate warmup+cosine LR over *total steps*.
- Gradient accumulation, EMA validation/checkpoint.
- Sigma-weighted loss; curriculum over noise.
- Optional OVERFIT mode to verify the pipeline can reach near-1.0 R² quickly.

Recommended Colab launch:
CUDA_VISIBLE_DEVICES=0 python train_latent_denoiser_steps.py \
  --data_dir /content/data \
  --out_dir /content/outputs \
  --device cuda \
  --batch_size 32 --grad_accum 2 \
  --total_steps 10000 --val_every 200 \
  --lr 3e-4 --warmup_steps 500 \
  --d_model 512 --ff_dim 2048 --num_layers 10 --nhead 8 \
  --noise_min 0.05 --noise_max 0.30 \
  --noise_min_init 0.02 --noise_max_init 0.15 \
  --target_r2 0.99
"""

import os, csv, math, argparse, gc, itertools, random
from pathlib import Path
from typing import List, Tuple, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


# -------------------- utilities --------------------

def read_shape_file(shape_path: Path) -> Tuple[int, int]:
    with open(shape_path, "r", encoding="utf-8") as f:
        kv = dict(ln.strip().split("=", 1) for ln in f if ln.strip())
    return int(kv["T"]), int(kv["F"])

def list_csvs(input_dir: Path, pattern_prefix: str = "_shard", pattern_suffix: str = ".csv") -> List[Path]:
    files = sorted([p for p in input_dir.glob(f"*{pattern_prefix}*{pattern_suffix}") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No CSV shards found under {input_dir}")
    return files

def read_row_to_latent(row: List[str], T: int, F: int) -> np.ndarray:
    vals = np.array([float(x) for x in row[1:]], dtype=np.float32)
    if vals.size != T * F:
        raise ValueError(f"Expected {T*F} latent values, got {vals.size}")
    return vals.reshape(T, F)

def compute_mean_std(csv_files: List[Path], shape_file: Path, max_rows: Optional[int] = 3000) -> Tuple[np.ndarray, np.ndarray]:
    T, F = read_shape_file(shape_file)
    count = 0
    mean = np.zeros((F,), dtype=np.float64)
    M2   = np.zeros((F,), dtype=np.float64)
    processed = 0
    for fp in csv_files:
        with open(fp, "r", encoding="utf-8") as fh:
            rdr = csv.reader(fh)
            next(rdr, None)
            for row in rdr:
                x = read_row_to_latent(row, T, F)
                xf = x.mean(axis=0)
                count += 1
                delta = xf - mean
                mean += delta / count
                M2 += delta * (xf - mean)
                processed += 1
                if (max_rows is not None) and (processed >= max_rows):
                    break
        if (max_rows is not None) and (processed >= max_rows):
            break
    var = M2 / max(1, count - 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


# -------------------- datasets --------------------

class LatentShardStream(IterableDataset):
    """
    Streams rows from CSV shards. Train/val split by modulo on a global counter.
    Yields (x_noisy_norm[T,F], x_clean_norm[T,F], sigma) with on-the-fly noise.
    """
    def __init__(
        self,
        csv_files: List[Path],
        shape_file: Path,
        split: str,                # "train" or "val"
        split_mod: int = 10,
        val_remainder: int = 0,
        noise_std_min: float = 0.05,
        noise_std_max: float = 0.30,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray]  = None,
        overfit_n_rows: int = 0,   # >0 => take first N rows only (both train and val use the same N with modulo split)
    ):
        super().__init__()
        assert split in ("train","val")
        self.csv_files = csv_files
        self.shape_file = shape_file
        self.split = split
        self.split_mod = split_mod
        self.val_remainder = val_remainder
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.T, self.F = read_shape_file(shape_file)
        self.mean = np.zeros((self.F,), dtype=np.float32) if mean is None else mean.astype(np.float32)
        self.std  = np.ones((self.F,), dtype=np.float32)  if std  is None else np.clip(std.astype(np.float32), 1e-6, None)
        self.overfit_n_rows = overfit_n_rows

        # Pre-materialize a list of rows if overfitting (tiny subset)
        self._overfit_rows = []
        if self.overfit_n_rows > 0:
            grabbed = 0
            for fp in self.csv_files:
                with open(fp, "r", encoding="utf-8") as fh:
                    rdr = csv.reader(fh)
                    next(rdr, None)
                    for row in rdr:
                        self._overfit_rows.append(row)
                        grabbed += 1
                        if grabbed >= self.overfit_n_rows:
                            break
                if grabbed >= self.overfit_n_rows:
                    break

    def _all_rows_iter(self) -> Iterable[Tuple[np.ndarray, float]]:
        gid = 0
        for fp in self.csv_files:
            with open(fp, "r", encoding="utf-8") as fh:
                rdr = csv.reader(fh)
                next(rdr, None)
                for row in rdr:
                    val_hit = (gid % self.split_mod) == self.val_remainder
                    if (self.split == "val" and not val_hit) or (self.split == "train" and val_hit):
                        gid += 1; continue
                    x = read_row_to_latent(row, self.T, self.F)
                    nl = np.random.rand() * (self.noise_std_max - self.noise_std_min) + self.noise_std_min
                    yield x, float(nl)
                    gid += 1

    def _overfit_rows_iter(self) -> Iterable[Tuple[np.ndarray, float]]:
        gid = 0
        for row in self._overfit_rows:
            val_hit = (gid % self.split_mod) == self.val_remainder
            if (self.split == "val" and not val_hit) or (self.split == "train" and val_hit):
                gid += 1; continue
            x = read_row_to_latent(row, self.T, self.F)
            nl = np.random.rand() * (self.noise_std_max - self.noise_std_min) + self.noise_std_min
            yield x, float(nl)
            gid += 1

    def __iter__(self):
        it = self._overfit_rows_iter() if self.overfit_n_rows > 0 else self._all_rows_iter()
        for x, nl in it:
            x_norm = (x - self.mean) / self.std
            eps = np.random.randn(*x_norm.shape).astype(np.float32)
            x_noisy = x_norm + nl * eps
            yield (
                torch.from_numpy(x_noisy),     # [T,F]
                torch.from_numpy(x_norm),      # [T,F]
                torch.tensor(nl, dtype=torch.float32),
            )


# -------------------- model --------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 128, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim; self.max_period = max_period
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period)*torch.arange(0, max(1,half), device=t.device) / max(1,half))
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class TransformerDenoiser(nn.Module):
    def __init__(self, feature_dim:int, seq_len:int, d_model:int=512, nhead:int=8, num_layers:int=10, dim_feedforward:int=2048, dropout:float=0.05, time_embed_dim:int=128):
        super().__init__()
        self.in_proj = nn.Linear(feature_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, d_model))
        self.t_embed = SinusoidalEmbedding(dim=time_embed_dim)
        self.t_proj = nn.Sequential(nn.Linear(time_embed_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, feature_dim)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, noisy_latents: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(noisy_latents)
        x = x + self.pos_embed.unsqueeze(0)
        t = self.t_proj(self.t_embed(noise_level))
        x = x + t.unsqueeze(1)
        h = self.encoder(x)
        eps_hat = self.out_proj(h)
        return eps_hat


# -------------------- EMA, metrics, schedules --------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)

@torch.no_grad()
def r2_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    err = y_true - y_pred
    sse = torch.sum(err * err)
    mu = torch.mean(y_true)
    sst = torch.sum((y_true - mu)**2) + 1e-12
    return float((1.0 - sse/sst).detach().cpu().item())

def make_noise_schedule(low_init, high_init, low_final, high_final, warm_steps=2000, ramp_steps=2000):
    """Returns function(step)->(lo,hi) with step-based curriculum."""
    def f(global_step):
        if global_step <= warm_steps:
            return low_init, high_init
        k = min(1.0, max(0.0, (global_step - warm_steps)/max(1, ramp_steps)))
        lo = low_init  + k*(low_final  - low_init)
        hi = high_init + k*(high_final - high_init)
        return lo, hi
    return f

def build_step_scheduler(opt, total_steps: int, warmup_steps: int):
    """Warmup + cosine over total_steps."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = min(1.0, max(0.0, t))
        return 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# -------------------- helpers --------------------

def infinite_loader(dataloader):
    """Cycle a DataLoader forever."""
    while True:
        for batch in dataloader:
            yield batch

def build_eval_model_like(model, device):
    """Create a new model with same dims as 'model'."""
    # extract dims from an attribute snapshot
    # (we kept them in constructor args originally; here infer from layers)
    m = model
    d_model = m.in_proj.out_features
    F = m.in_proj.in_features
    T = m.pos_embed.shape[0]
    # retrieve encoder config (we can’t read num_layers directly; store in attr)
    num_layers = len(m.encoder.layers)
    nhead = m.encoder.layers[0].self_attn.num_heads
    ff = m.encoder.layers[0].linear1.out_features
    dropout = m.encoder.layers[0].dropout.p
    time_dim = m.t_proj[0].in_features
    eval_model = TransformerDenoiser(F, T, d_model, nhead, num_layers, ff, dropout, time_dim).to(device).eval()
    return eval_model


# -------------------- train --------------------

def train(args):
    # device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if (torch.cuda.is_available() and torch.version.cuda is not None and torch.cuda.device_count() > 0 and not args.force_cpu) else "cpu"
    print(f"[Device] {device}")

    # paths & data
    data_dir = Path(args.data_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list_csvs(data_dir)
    shape_file = data_dir / "latents_shape.txt"
    T, F = read_shape_file(shape_file)
    print(f"[Data] shards={len(csv_files)}  T={T}  F={F}")

    # stats
    print(f"[Stats] Computing mean/std over ~{args.stats_rows or 'ALL'} rows…")
    mean, std = compute_mean_std(csv_files, shape_file, max_rows=(None if args.stats_rows <= 0 else args.stats_rows))
    np.save(out_dir / "mean.npy", mean); np.save(out_dir / "std.npy", std)
    print(f"[Stats] Saved mean/std to {out_dir}")

    # model/opt/ema
    model = TransformerDenoiser(
        feature_dim=F, seq_len=T,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.ff_dim, dropout=args.dropout, time_embed_dim=args.time_embed_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = GradScaler(enabled=(device=="cuda"))
    ema = EMA(model, decay=args.ema_decay)

    # curriculum and scheduler (step-based)
    noise_sched = make_noise_schedule(
        args.noise_min_init, args.noise_max_init,
        args.noise_min,      args.noise_max,
        warm_steps=args.curr_warm_steps, ramp_steps=args.curr_ramp_steps,
    )
    sched = build_step_scheduler(opt, total_steps=args.total_steps, warmup_steps=args.warmup_steps)

    # loaders (small host footprint)
    train_ds = LatentShardStream(csv_files, shape_file, split="train",
                                 split_mod=args.val_mod, val_remainder=args.val_remainder,
                                 noise_std_min=args.noise_min, noise_std_max=args.noise_max,
                                 mean=mean, std=std, overfit_n_rows=args.overfit_n_rows)
    val_ds   = LatentShardStream(csv_files, shape_file, split="val",
                                 split_mod=args.val_mod, val_remainder=args.val_remainder,
                                 noise_std_min=args.noise_min, noise_std_max=args.noise_max,
                                 mean=mean, std=std, overfit_n_rows=args.overfit_n_rows)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   **loader_kwargs)

    train_iter = infinite_loader(train_loader)

    # training loop over steps
    best_r2 = -1.0
    global_step = 0
    accum = max(1, args.grad_accum)
    opt.zero_grad(set_to_none=True)

    while global_step < args.total_steps:
        # update curriculum range at this step
        lo, hi = noise_sched(global_step)
        # temporarily adjust dataset noise range (easy: replace fields)
        train_ds.noise_std_min, train_ds.noise_std_max = lo, hi
        val_ds.noise_std_min,   val_ds.noise_std_max   = lo, hi

        x_noisy, x_clean, nl = next(train_iter)
        x_noisy = x_noisy.to(device, non_blocking=False)
        x_clean = x_clean.to(device, non_blocking=False)
        nl = nl.to(device, non_blocking=False)

        residual_target = x_noisy - x_clean

        with autocast(enabled=(device=="cuda")):
            eps_hat = model(x_noisy, nl)
            c = args.sigma_weight_c
            w = (nl * nl) / (nl * nl + c)      # [B]
            w = w.view(-1, 1, 1)
            loss = torch.mean(w * (eps_hat - residual_target)**2) / accum

        scaler.scale(loss).backward()

        if (global_step + 1) % accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            ema.update(model)
            sched.step()

        global_step += 1

        if global_step % args.log_every == 0:
            print(f"[Step {global_step:06d}] loss={float(loss.detach().cpu().item())*accum:.6f} | "
                  f"noise=[{lo:.3f},{hi:.3f}] | lr={opt.param_groups[0]['lr']:.2e}")

        # validation
        if (global_step % args.val_every == 0) or (global_step == args.total_steps):
            model.eval()
            with torch.no_grad():
                eval_model = build_eval_model_like(model, device)
                ema.copy_to(eval_model); eval_model.eval()
                r2_vals = []
                for x_noisy_v, x_clean_v, nl_v in val_loader:
                    x_noisy_v = x_noisy_v.to(device, non_blocking=False)
                    x_clean_v = x_clean_v.to(device, non_blocking=False)
                    nl_v = nl_v.to(device, non_blocking=False)
                    with autocast(enabled=(device=="cuda")):
                        eps_hat_v = eval_model(x_noisy_v, nl_v)
                        xhat_v = x_noisy_v - eps_hat_v
                    r2_vals.append(r2_metric(x_clean_v, xhat_v))
                val_r2 = float(np.mean(r2_vals)) if r2_vals else -1.0
            del eval_model
            if device == "cuda": torch.cuda.empty_cache()
            gc.collect()

            print(f"[Val @ step {global_step}] R²(EMA)={val_r2:.4f}  (best={best_r2:.4f})")

            if val_r2 > best_r2:
                best_r2 = val_r2
                # save EMA weights
                ckpt_model = build_eval_model_like(model, "cpu").eval()
                ema.copy_to(ckpt_model)
                ckpt = {
                    "model": ckpt_model.state_dict(),
                    "meta": {
                        "T": T, "F": F,
                        "d_model": model.in_proj.out_features,
                        "nhead": model.encoder.layers[0].self_attn.num_heads,
                        "num_layers": len(model.encoder.layers),
                        "ff_dim": model.encoder.layers[0].linear1.out_features,
                        "dropout": model.encoder.layers[0].dropout.p,
                        "time_embed_dim": model.t_proj[0].in_features,
                        "noise_min": args.noise_min, "noise_max": args.noise_max,
                        "mean_path": str(out_dir / "mean.npy"),
                        "std_path": str(out_dir / "std.npy"),
                    }
                }
                torch.save(ckpt, Path(args.out_dir) / "latent_denoiser_best.pt")
                del ckpt_model; gc.collect()
                print(f"[Checkpoint] Saved EMA model at step {global_step} with R²={best_r2:.4f}")

            model.train()

        if best_r2 >= args.target_r2:
            print(f"[Done] Target R² {args.target_r2:.2f}+ reached at step {global_step}. Best={best_r2:.4f}")
            break

    print(f"[Summary] Best val R² (EMA): {best_r2:.4f}")
    print(f"[Artifacts] {args.out_dir}")


# -------------------- optional viz --------------------

def viz_eval(args):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Matplotlib not available; skipping viz.", e); return
    data_dir = Path(args.data_dir); out_dir = Path(args.out_dir)
    csv_files = list_csvs(data_dir); shape_file = data_dir / "latents_shape.txt"
    T, F = read_shape_file(shape_file)
    mean = np.load(out_dir / "mean.npy"); std = np.load(out_dir / "std.npy")
    ckpt = torch.load(Path(out_dir) / "latent_denoiser_best.pt", map_location="cpu")
    meta = ckpt["meta"]
    model = TransformerDenoiser(F, T, meta["d_model"], meta["nhead"], meta["num_layers"], meta["ff_dim"], meta["dropout"], meta["time_embed_dim"]).eval()
    model.load_state_dict(ckpt["model"], strict=False)

    def sample_rows(files: List[Path], n=500):
        rows = []
        per = max(1, n // len(files))
        for fp in files:
            with open(fp, "r", encoding="utf-8") as fh:
                rdr = csv.reader(fh); next(rdr, None)
                for i, row in enumerate(rdr):
                    if i >= per: break
                    rows.append(row)
        return rows

    rows = sample_rows(csv_files, n=500)
    bins = np.linspace(max(0.01,args.noise_min_init), args.noise_max, 8)
    bin_r2 = [[] for _ in range(len(bins)-1)]
    feat_sse = np.zeros((F,), dtype=np.float64); feat_sst = np.zeros((F,), dtype=np.float64)

    for row in rows:
        x = read_row_to_latent(row, T, F)
        x_norm = (x - mean) / np.clip(std, 1e-6, None)
        for nl in [0.05, 0.10, 0.20, 0.30]:
            eps = np.random.randn(*x_norm.shape).astype(np.float32)
            xn = x_norm + nl * eps
            xt = torch.from_numpy(x_norm).unsqueeze(0)
            xn_t = torch.from_numpy(xn).unsqueeze(0)
            nt = torch.tensor([nl], dtype=torch.float32)
            with torch.no_grad():
                eps_hat = model(xn_t, nt)
                xhat = xn_t - eps_hat
            r2 = r2_metric(xt, xhat)
            b = int(np.clip(np.digitize([nl], bins) - 1, 0, len(bins)-2))
            bin_r2[b].append(r2)
            diff = (xt - xhat).squeeze(0).numpy()
            mu_f = xt.squeeze(0).numpy().mean(axis=0)
            feat_sse += (diff**2).sum(axis=0)
            feat_sst += ((xt.squeeze(0).numpy() - mu_f)**2).sum(axis=0)

    # plots
    bin_centers = 0.5*(bins[:-1]+bins[1:]); avg_r2 = [np.mean(v) if v else np.nan for v in bin_r2]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4)); plt.plot(bin_centers, avg_r2, marker="o"); plt.ylim(0.9, 1.0); plt.grid(True)
    plt.xlabel("Noise σ"); plt.ylabel("R²"); plt.title("R² vs noise"); plt.show()

    feat_r2 = 1.0 - feat_sse / (feat_sst + 1e-12)
    plt.figure(figsize=(6,4)); plt.hist(feat_r2, bins=40)
    plt.xlabel("Per-feature R²"); plt.ylabel("Count"); plt.title("Per-feature R² distribution")
    plt.show()


# -------------------- CLI --------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FlowMo latent denoiser (step-based trainer)")

    # Paths
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir",  type=str, required=True)

    # Device
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--force_cpu", action="store_true")

    # Model / train
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=10)
    ap.add_argument("--ff_dim", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--time_embed_dim", type=int, default=128)
    ap.add_argument("--ema_decay", type=float, default=0.999)

    # Steps & eval frequency
    ap.add_argument("--total_steps", type=int, default=10000)
    ap.add_argument("--val_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--warmup_steps", type=int, default=500)

    # Noise ranges (final + curriculum in *steps*)
    ap.add_argument("--noise_min", type=float, default=0.05)
    ap.add_argument("--noise_max", type=float, default=0.30)
    ap.add_argument("--noise_min_init", type=float, default=0.02)
    ap.add_argument("--noise_max_init", type=float, default=0.15)
    ap.add_argument("--curr_warm_steps", type=int, default=2000)
    ap.add_argument("--curr_ramp_steps", type=int, default=2000)

    # Split / stats
    ap.add_argument("--val_mod", type=int, default=10)
    ap.add_argument("--val_remainder", type=int, default=0)
    ap.add_argument("--stats_rows", type=int, default=3000)

    # Loss weighting
    ap.add_argument("--sigma_weight_c", type=float, default=0.0025)

    # Targets & overfit debug
    ap.add_argument("--target_r2", type=float, default=0.99)
    ap.add_argument("--overfit_n_rows", type=int, default=0, help=">0 to overfit on N rows as a sanity check")

    # Viz
    ap.add_argument("--viz_after", action="store_true")

    args = ap.parse_args()
    train(args)
    if args.viz_after:
        viz_eval(args)
