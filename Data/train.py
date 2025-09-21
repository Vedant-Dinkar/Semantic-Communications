# -*- coding: utf-8 -*-
"""
FlowMo Latent Denoiser (Residual training, Curriculum, AMP, EMA, Step-accurate LR)
- Reads CSV shards + latents_shape.txt (T, F)
- Streams data; feature-wise z-score
- Predicts residual epsilon; reconstructs x̂ = x_noisy - ε̂
- Accurate LR schedule (warmup+cosine) using true steps/epoch
- Gradient accumulation, EMA validation/checkpoint
- Optional sigma-weighted loss; curriculum on noise
- Optional quick visualization post-training

Run (Colab, prefers subprocess):
CUDA_VISIBLE_DEVICES=0 python train_latent_denoiser_full.py \
  --data_dir /content/data \
  --out_dir /content/outputs \
  --device cuda \
  --batch_size 32 --grad_accum 2 \
  --epochs 20 --lr 3e-4 --warmup_steps 200 \
  --d_model 512 --ff_dim 2048 --num_layers 10 --nhead 8 \
  --noise_min 0.05 --noise_max 0.30 \
  --noise_min_init 0.02 --noise_max_init 0.15 \
  --curr_warm_epochs 3 --curr_ramp_epochs 3 \
  --target_r2 0.99
"""

import os, csv, math, argparse, gc
from pathlib import Path
from typing import List, Tuple, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


# ============ Basic file utilities ============

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
    return vals.reshape(T, F)  # [T,F]

def compute_mean_std(csv_files: List[Path], shape_file: Path, max_rows: Optional[int] = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """Feature-wise stats over per-row token-averages (Welford)."""
    T, F = read_shape_file(shape_file)
    count = 0
    mean = np.zeros((F,), dtype=np.float64)
    M2   = np.zeros((F,), dtype=np.float64)
    processed = 0

    for fp in csv_files:
        with open(fp, "r", encoding="utf-8") as fh:
            rdr = csv.reader(fh)
            next(rdr, None)  # header
            for row in rdr:
                x = read_row_to_latent(row, T, F)  # [T,F]
                xf = x.mean(axis=0)                # per-feature mean across tokens
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


# ============ Streaming dataset (train/val split by modulo) ============

class LatentShardStream(IterableDataset):
    """
    Streams rows from CSV shards. Validation split via
    (global_idx % split_mod) == val_remainder.

    Emits normalized noisy latents, clean latents, and noise level:
      (x_noisy_norm[T,F], x_clean_norm[T,F], sigma)
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
        mean: Optional[np.ndarray] = None,   # [F]
        std: Optional[np.ndarray]  = None,   # [F]
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

    def _row_iter(self) -> Iterable[Tuple[np.ndarray, float]]:
        gid = 0
        for fp in self.csv_files:
            with open(fp, "r", encoding="utf-8") as fh:
                rdr = csv.reader(fh)
                next(rdr, None)
                for row in rdr:
                    val_hit = (gid % self.split_mod) == self.val_remainder
                    if (self.split == "val" and not val_hit) or (self.split == "train" and val_hit):
                        gid += 1
                        continue
                    x = read_row_to_latent(row, self.T, self.F)
                    nl = np.random.rand() * (self.noise_std_max - self.noise_std_min) + self.noise_std_min
                    yield x, float(nl)
                    gid += 1

    def __iter__(self):
        for x, nl in self._row_iter():
            x_norm = (x - self.mean) / self.std
            eps = np.random.randn(*x_norm.shape).astype(np.float32)
            x_noisy = x_norm + nl * eps
            yield (
                torch.from_numpy(x_noisy),     # [T,F]
                torch.from_numpy(x_norm),      # [T,F]
                torch.tensor(nl, dtype=torch.float32),
            )


# ============ Model ============

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 128, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period)*torch.arange(0, half, device=t.device) / max(1, half))
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class TransformerDenoiser(nn.Module):
    """Token-wise transformer that predicts residual ε ≈ x_noisy − x_clean."""
    def __init__(
        self,
        feature_dim: int,   # F
        seq_len: int,       # T
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 10,
        dim_feedforward: int = 2048,
        dropout: float = 0.05,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        self.in_proj = nn.Linear(feature_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, d_model))

        self.t_embed = SinusoidalEmbedding(dim=time_embed_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, feature_dim)

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, noisy_latents: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        B, T, F = noisy_latents.shape
        x = self.in_proj(noisy_latents)                # [B,T,d]
        x = x + self.pos_embed.unsqueeze(0)            # learned pos enc
        t = self.t_proj(self.t_embed(noise_level))     # [B,d]
        x = x + t.unsqueeze(1)
        h = self.encoder(x)                            # [B,T,d]
        eps_hat = self.out_proj(h)                     # [B,T,F]
        return eps_hat


# ============ EMA & Metrics ============

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


# ============ Curriculum & Scheduler helpers ============

def make_noise_schedule(low_init, high_init, low_final, high_final, warm_epochs=3, ramp_epochs=3):
    """Returns a function(epoch)->(low, high)."""
    def f(ep):
        if ep <= warm_epochs:
            return low_init, high_init
        k = min(1.0, max(0.0, (ep - warm_epochs)/max(1, ramp_epochs)))
        lo = low_init  + k*(low_final  - low_init)
        hi = high_init + k*(high_final - high_init)
        return lo, hi
    return f

def build_scheduler(opt, steps_per_epoch: int, epoch_idx: int, warmup_steps_cfg: int, rem_epochs: int):
    """Step-accurate warmup+cosine over the remaining training horizon."""
    total_steps = max(1, steps_per_epoch * rem_epochs)
    warmup = min(warmup_steps_cfg, max(1, total_steps // 10))  # cap warmup to 10% of remaining

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        t = min(1.0, max(0.0, t))
        return 0.5 * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ============ Train ============

def train(args):
    # Device selection
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if (torch.cuda.is_available() and torch.version.cuda is not None and torch.cuda.device_count() > 0 and not args.force_cpu) else "cpu"
    print(f"[Device] {device}")

    data_dir = Path(args.data_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list_csvs(data_dir)
    shape_file = data_dir / "latents_shape.txt"
    T, F = read_shape_file(shape_file)
    print(f"[Data] shards={len(csv_files)}  T={T}  F={F}")

    # Stats
    print(f"[Stats] Computing mean/std over ~{args.stats_rows or 'ALL'} rows…")
    mean, std = compute_mean_std(csv_files, shape_file, max_rows=(None if args.stats_rows <= 0 else args.stats_rows))
    np.save(out_dir / "mean.npy", mean); np.save(out_dir / "std.npy", std)
    print(f"[Stats] Saved mean/std to {out_dir}")

    # Model / Opt / AMP / EMA
    model = TransformerDenoiser(
        feature_dim=F, seq_len=T,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.ff_dim, dropout=args.dropout, time_embed_dim=args.time_embed_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = GradScaler(enabled=(device=="cuda"))
    ema = EMA(model, decay=args.ema_decay)

    # Curriculum schedule
    noise_sched = make_noise_schedule(
        args.noise_min_init, args.noise_max_init,
        args.noise_min,      args.noise_max,
        warm_epochs=args.curr_warm_epochs,
        ramp_epochs=args.curr_ramp_epochs,
    )

    best_r2 = -1.0
    patience_hits = 0

    for epoch in range(1, args.epochs + 1):
        lo, hi = noise_sched(epoch)

        # Rebuild loaders each epoch for the current noise range
        train_ds = LatentShardStream(csv_files, shape_file, split="train",
                                     split_mod=args.val_mod, val_remainder=args.val_remainder,
                                     noise_std_min=lo, noise_std_max=hi, mean=mean, std=std)
        val_ds   = LatentShardStream(csv_files, shape_file, split="val",
                                     split_mod=args.val_mod, val_remainder=args.val_remainder,
                                     noise_std_min=lo, noise_std_max=hi, mean=mean, std=std)
        loader_kwargs = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False)
        train_loader = DataLoader(train_ds, **loader_kwargs)
        val_loader   = DataLoader(val_ds,   **loader_kwargs)

        def count_rows(csv_files, split, split_mod, val_remainder):
          gid, count = 0, 0
          for fp in csv_files:
            with open(fp, "r", encoding="utf-8") as fh:
              rdr = csv.reader(fh)
              next(rdr, None)
              for _ in rdr:
                val_hit = (gid % split_mod) == val_remainder
                if (split == "val" and val_hit) or (split == "train" and not val_hit):
                    count += 1
                gid += 1
          return count

        train_rows = count_rows(csv_files, "train", args.val_mod, args.val_remainder)
        steps_per_epoch = max(1, train_rows // (args.batch_size * max(1, args.grad_accum)))
        
        rem_epochs = args.epochs - epoch + 1
        sched = build_scheduler(opt, steps_per_epoch, epoch, args.warmup_steps, rem_epochs)

        # ---- Train ----
        model.train()
        total_loss, batches = 0.0, 0
        accum = max(1, args.grad_accum)
        opt.zero_grad(set_to_none=True)

        for step, (x_noisy, x_clean, nl) in enumerate(train_loader, start=1):
            x_noisy = x_noisy.to(device, non_blocking=False)   # [B,T,F]
            x_clean = x_clean.to(device, non_blocking=False)
            nl = nl.to(device, non_blocking=False)              # [B]

            residual_target = x_noisy - x_clean                 # ε = x_noisy - x_clean

            with autocast(enabled=(device=="cuda")):
                eps_hat = model(x_noisy, nl)
                # Sigma-weighted loss (stabilizes high-σ)
                c = args.sigma_weight_c
                w = (nl * nl) / (nl * nl + c)                   # [B]
                w = w.view(-1, 1, 1)
                loss = torch.mean(w * (eps_hat - residual_target)**2) / accum

            scaler.scale(loss).backward()

            if step % accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
                sched.step()

            total_loss += float(loss.detach().cpu().item()) * accum
            batches += 1

        train_mse = total_loss / max(1, batches)

        # ---- Validate with EMA weights (no cloning big state) ----
        model.eval()
        with torch.no_grad():
            # build temporary eval model; load EMA weights
            eval_model = TransformerDenoiser(
                feature_dim=F, seq_len=T,
                d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
                dim_feedforward=args.ff_dim, dropout=args.dropout, time_embed_dim=args.time_embed_dim,
            ).to(device).eval()
            ema.copy_to(eval_model)

            r2_vals = []
            for x_noisy, x_clean, nl in val_loader:
                x_noisy = x_noisy.to(device, non_blocking=False)
                x_clean = x_clean.to(device, non_blocking=False)
                nl = nl.to(device, non_blocking=False)
                with autocast(enabled=(device=="cuda")):
                    eps_hat = eval_model(x_noisy, nl)
                    xhat = x_noisy - eps_hat
                r2_vals.append(r2_metric(x_clean, xhat))
            val_r2 = float(np.mean(r2_vals)) if r2_vals else -1.0

        # free eval model
        del eval_model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        print(f"[Epoch {epoch:02d}] noise=[{lo:.3f},{hi:.3f}] | steps/epoch={steps_per_epoch} | "
              f"train_mse={train_mse:.6f} | val_R2(EMA)={val_r2:.4f} | lr={opt.param_groups[0]['lr']:.2e}")

        # Save best EMA checkpoint
        if val_r2 > best_r2:
            best_r2 = val_r2
            tmp_eval = TransformerDenoiser(
                feature_dim=F, seq_len=T,
                d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
                dim_feedforward=args.ff_dim, dropout=args.dropout, time_embed_dim=args.time_embed_dim,
            ).to("cpu").eval()
            ema.copy_to(tmp_eval)
            ckpt = {
                "model": tmp_eval.state_dict(),
                "meta": {
                    "T": T, "F": F,
                    "d_model": args.d_model, "nhead": args.nhead, "num_layers": args.num_layers,
                    "ff_dim": args.ff_dim, "dropout": args.dropout, "time_embed_dim": args.time_embed_dim,
                    "noise_min": args.noise_min, "noise_max": args.noise_max,
                    "mean_path": str(out_dir / "mean.npy"),
                    "std_path": str(out_dir / "std.npy"),
                }
            }
            torch.save(ckpt, out_dir / "latent_denoiser_best.pt")
            del tmp_eval; gc.collect()
            print(f"[Checkpoint] Saved best EMA model with R²={best_r2:.4f}")

        # Early stop
        if best_r2 >= args.target_r2:
            print(f"[Done] Target R² {args.target_r2:.2f}+ reached. Best={best_r2:.4f}")
            break

        if epoch > 1 and val_r2 < best_r2 - 1e-4:
            patience_hits += 1
            if patience_hits >= args.patience:
                print(f"[Stop] Early stop, no improvement. Best R²={best_r2:.4f}")
                break

    print(f"[Summary] Best val R² (EMA): {best_r2:.4f}")
    print(f"[Artifacts] {out_dir}")


# ============ Optional: quick viz after training ============

def viz_eval(args):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Matplotlib not available; skipping viz.", e)
        return

    data_dir = Path(args.data_dir); out_dir = Path(args.out_dir)
    csv_files = list_csvs(data_dir)
    shape_file = data_dir / "latents_shape.txt"
    T, F = read_shape_file(shape_file)

    mean = np.load(out_dir / "mean.npy"); std = np.load(out_dir / "std.npy")
    ckpt = torch.load(out_dir / "latent_denoiser_best.pt", map_location="cpu")
    meta = ckpt["meta"]

    model = TransformerDenoiser(
        feature_dim=F, seq_len=T,
        d_model=meta["d_model"], nhead=meta["nhead"], num_layers=meta["num_layers"],
        dim_feedforward=meta["ff_dim"], dropout=meta["dropout"], time_embed_dim=meta["time_embed_dim"],
    ).eval()
    model.load_state_dict(ckpt["model"], strict=False)

    # sample a small set for viz
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
            # overall R²
            r2 = r2_metric(xt, xhat)
            b = int(np.clip(np.digitize([nl], bins) - 1, 0, len(bins)-2))
            bin_r2[b].append(r2)
            # per-feature accum
            diff = (xt - xhat).squeeze(0).numpy()
            mu_f = xt.squeeze(0).numpy().mean(axis=0)
            feat_sse += (diff**2).sum(axis=0)
            feat_sst += ((xt.squeeze(0).numpy() - mu_f)**2).sum(axis=0)

    # R² vs noise
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    avg_r2 = [np.mean(v) if v else np.nan for v in bin_r2]
    plt.figure(figsize=(6,4)); plt.plot(bin_centers, avg_r2, marker="o")
    plt.ylim(0.9, 1.0); plt.grid(True)
    plt.xlabel("Noise σ"); plt.ylabel("R²"); plt.title("R² vs noise")
    plt.show()

    # Per-feature R² distribution
    feat_r2 = 1.0 - feat_sse / (feat_sst + 1e-12)
    plt.figure(figsize=(6,4)); plt.hist(feat_r2, bins=40)
    plt.xlabel("Per-feature R²"); plt.ylabel("Count"); plt.title("Per-feature R² distribution")
    plt.show()


# ============ CLI ============

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FlowMo latent denoiser training")

    # Paths
    ap.add_argument("--data_dir", type=str, required=True, help="Folder with CSV shards + latents_shape.txt")
    ap.add_argument("--out_dir",  type=str, required=True, help="Folder to save checkpoints & stats")

    # Device
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"], help="Override device")
    ap.add_argument("--force_cpu", action="store_true")

    # Model / train
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    ap.add_argument("--epochs", type=int, default=20)
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

    # Noise ranges (final + curriculum)
    ap.add_argument("--noise_min", type=float, default=0.05)
    ap.add_argument("--noise_max", type=float, default=0.30)
    ap.add_argument("--noise_min_init", type=float, default=0.02)
    ap.add_argument("--noise_max_init", type=float, default=0.15)
    ap.add_argument("--curr_warm_epochs", type=int, default=3)
    ap.add_argument("--curr_ramp_epochs", type=int, default=3)

    # Split / stats
    ap.add_argument("--val_mod", type=int, default=10)
    ap.add_argument("--val_remainder", type=int, default=0)
    ap.add_argument("--stats_rows", type=int, default=3000)

    # Scheduler
    ap.add_argument("--warmup_steps", type=int, default=200)

    # Loss weighting
    ap.add_argument("--sigma_weight_c", type=float, default=0.0025, help="w = σ²/(σ²+c)")

    # Targets / runtime
    ap.add_argument("--target_r2", type=float, default=0.99)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--viz_after", action="store_true")

    args = ap.parse_args()

    train(args)
    if args.viz_after:
        viz_eval(args)
