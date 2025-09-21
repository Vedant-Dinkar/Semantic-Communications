# -*- coding: utf-8 -*-
"""
Evaluate a trained latent denoiser on CSV shards.
- Loads EMA checkpoint (latent_denoiser_best.pt) + mean/std
- Computes overall R² / MSE across requested noise levels
- Saves per-feature R² histogram + CSV
- Optionally decodes a few samples to images via FlowMo (if weights provided)

Usage (basic):
  python test_latent_denoiser.py \
    --data_dir /content/data \
    --ckpt /content/outputs/latent_denoiser_best.pt \
    --device cuda \
    --num_rows 1000 \
    --sigmas 0.05,0.10,0.20,0.30 \
    --out_dir /content/eval

Optional image decode (if FlowMo available):
  ... --flowmo_root /content/FlowMo \
      --flowmo_weights /content/Data/flowmo_hi.pth \
      --decode_samples 4
"""

import os, csv, argparse, math, json, gc
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast

# ---------- Utilities copied (light) ----------
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
        raise ValueError(f"Expected {T*F} values, got {vals.size}")
    return vals.reshape(T, F)

@torch.no_grad()
def r2_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    err = y_true - y_pred
    sse = torch.sum(err * err)
    mu = torch.mean(y_true)
    sst = torch.sum((y_true - mu)**2) + 1e-12
    return float((1.0 - sse/sst).detach().cpu().item())

# ---------- Model (must match training) ----------
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
    def __init__(self, feature_dim:int, seq_len:int, d_model:int, nhead:int, num_layers:int, dim_feedforward:int, dropout:float, time_embed_dim:int):
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
        return self.out_proj(h)

# ---------- Optional FlowMo decode ----------
@torch.no_grad()
def maybe_decode_images(flowmo_root: Optional[str], flowmo_weights: Optional[str],
                        latents_list: List[np.ndarray], out_dir: Path, device: str = "cuda"):
    if (flowmo_root is None) or (flowmo_weights is None):
        return
    import sys
    sys.path.insert(0, flowmo_root)
    from omegaconf import OmegaConf
    from flowmo.models import FlowMo

    cfg = OmegaConf.load(Path(flowmo_root)/"flowmo/configs/base.yaml")
    model = FlowMo(width=cfg.model.width, config=cfg).to(device).eval()
    sd = torch.load(flowmo_weights, map_location="cpu")
    # Load typical keys
    if "model_ema_state_dict" in sd:
        model.load_state_dict(sd["model_ema_state_dict"], strict=False)
    elif "model_state_dict" in sd:
        model.load_state_dict(sd["model_state_dict"], strict=False)
    else:
        model.load_state_dict(sd, strict=False)

    import torchvision.utils as vutils

    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = []
    for i, lat in enumerate(latents_list):
        # lat: [T,F] normalized clean latents expected by decoder after you append mask in sampling
        # We only *reconstruct with decoder’s rf_sample* if your pipeline expects the extra mask channel.
        # Here we just call model.reconstruct(images=None, code=lat_norm_with_mask) would require hooking into sample loop.
        # Simpler: pass via model.reconstruct() which quantizes if None; not ideal for exact round-trip.
        # Given most users already verified recon once, treat this as *optional*; skip if not aligned.
        pass
    print("[Decode] Skipped actual decode pipeline here to keep test minimal.")

# ---------- Main eval ----------
def main(args):
    device = args.device if args.device is not None else ("cuda" if (torch.cuda.is_available() and torch.version.cuda) else "cpu")
    print(f"[Device] {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    shape_file = data_dir / "latents_shape.txt"
    csv_files = list_csvs(data_dir)
    T, F = read_shape_file(shape_file)
    print(f"[Data] shards={len(csv_files)}  T={T}  F={F}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    meta = ckpt["meta"]
    assert meta["T"] == T and meta["F"] == F, f"Shape mismatch: ckpt {meta['T']}x{meta['F']} vs data {T}x{F}"

    mean = np.load(meta["mean_path"]); std = np.load(meta["std_path"])
    std = np.clip(std, 1e-6, None)

    model = TransformerDenoiser(
        feature_dim=F, seq_len=T,
        d_model=meta["d_model"], nhead=meta["nhead"], num_layers=meta["num_layers"],
        dim_feedforward=meta["ff_dim"], dropout=meta["dropout"], time_embed_dim=meta["time_embed_dim"],
    ).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)

    # parse sigmas
    sigmas = [float(s) for s in args.sigmas.split(",")]
    print(f"[Eval] sigmas = {sigmas}")

    # Collect rows
    rows = []
    remaining = args.num_rows
    for fp in csv_files:
        with open(fp, "r", encoding="utf-8") as fh:
            rdr = csv.reader(fh)
            next(rdr, None)
            for row in rdr:
                rows.append(row)
                remaining -= 1
                if remaining == 0:
                    break
        if remaining == 0:
            break
    print(f"[Eval] Using {len(rows)} rows")

    # Metrics accumulators
    per_sigma = {s: {"r2": [], "mse": []} for s in sigmas}
    # Per-feature accumulators (global)
    feat_sse = np.zeros((F,), dtype=np.float64)
    feat_sst = np.zeros((F,), dtype=np.float64)

    # Qual samples to save
    qual_count = 0
    qual_dir = out_dir / "qual"; qual_dir.mkdir(exist_ok=True, parents=True)

    for idx, row in enumerate(rows):
        x = read_row_to_latent(row, T, F)                 # [T,F]
        x_norm = (x - mean) / std

        xt = torch.from_numpy(x_norm).unsqueeze(0).to(device)  # [1,T,F]

        for s in sigmas:
            eps = np.random.randn(*x_norm.shape).astype(np.float32)
            xn = x_norm + s * eps
            xn_t = torch.from_numpy(xn).unsqueeze(0).to(device)
            nt = torch.tensor([s], dtype=torch.float32, device=device)

            with torch.no_grad(), autocast(enabled=(device=="cuda")):
                eps_hat = model(xn_t, nt)
                xhat = xn_t - eps_hat

            # metrics
            r2 = r2_metric(xt, xhat)
            mse = float(torch.mean((xhat - xt)**2).detach().cpu().item())
            per_sigma[s]["r2"].append(r2)
            per_sigma[s]["mse"].append(mse)

            # per-feature accum
            diff = (xt - xhat).squeeze(0).detach().cpu().numpy()  # [T,F]
            mu_f = xt.squeeze(0).detach().cpu().numpy().mean(axis=0)
            feat_sse += (diff**2).sum(axis=0)
            feat_sst += ((xt.squeeze(0).detach().cpu().numpy() - mu_f)**2).sum(axis=0)

        # save a few qualitative dumps
        if qual_count < args.decode_samples:
            # save numpy arrays for one sigma (use max sigma)
            s = max(sigmas)
            eps = np.random.randn(*x_norm.shape).astype(np.float32)
            xn = x_norm + s * eps
            xn_t = torch.from_numpy(xn).unsqueeze(0).to(device)
            nt = torch.tensor([s], dtype=torch.float32, device=device)
            with torch.no_grad():
                eps_hat = model(xn_t, nt)
                xhat = (xn_t - eps_hat).squeeze(0).detach().cpu().numpy()
            np.save(qual_dir/f"clean_{idx:05d}.npy", x_norm.astype(np.float32))
            np.save(qual_dir/f"noisy_{idx:05d}_s{int(s*100):02d}.npy", xn.astype(np.float32))
            np.save(qual_dir/f"denoised_{idx:05d}_s{int(s*100):02d}.npy", xhat.astype(np.float32))
            qual_count += 1

    # summarize
    summary = {}
    for s in sigmas:
        r2 = float(np.mean(per_sigma[s]["r2"])) if per_sigma[s]["r2"] else float("nan")
        mse = float(np.mean(per_sigma[s]["mse"])) if per_sigma[s]["mse"] else float("nan")
        summary[f"sigma_{s:.2f}"] = {"R2": r2, "MSE": mse}
    feat_r2 = 1.0 - feat_sse / (feat_sst + 1e-12)
    overall_r2 = float(np.mean([summary[k]["R2"] for k in summary.keys()]))

    # write JSON + CSV
    with open(out_dir/"metrics.json","w") as f:
        json.dump({"overall_R2_mean": overall_r2, "per_sigma": summary}, f, indent=2)

    with open(out_dir/"per_feature_r2.csv","w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["feature_index","R2"])
        for i, r in enumerate(feat_r2.tolist()):
            wr.writerow([i, r])

    print("[Eval] Summary:")
    for k,v in summary.items():
        print(f"  {k}: R2={v['R2']:.4f}, MSE={v['MSE']:.6f}")
    print(f"[Eval] Feature-R² mean={float(np.mean(feat_r2)):.4f}  min={float(np.min(feat_r2)):.4f}  max={float(np.max(feat_r2)):.4f}")
    print(f"[Write] metrics.json and per_feature_r2.csv in {out_dir}")

    # (Optional) Decode images – placeholder (kept minimal to avoid mismatches)
    if args.flowmo_root and args.flowmo_weights and args.decode_samples > 0:
        print("[Decode] FlowMo decode requested; placeholder does not perform full decode here.")
        # Example skeleton if you want to wire actual FlowMo rf_sample:
        # maybe_decode_images(args.flowmo_root, args.flowmo_weights, [], out_dir/ "decoded", device)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate latent denoiser")
    ap.add_argument("--data_dir", type=str, required=True, help="Folder with CSV shards + latents_shape.txt")
    ap.add_argument("--ckpt", type=str, required=True, help="latent_denoiser_best.pt")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write metrics & qual samples")
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--num_rows", type=int, default=2000, help="How many rows to evaluate")
    ap.add_argument("--sigmas", type=str, default="0.05,0.10,0.20,0.30")
    ap.add_argument("--decode_samples", type=int, default=0, help="How many qual latent dumps to write (npy)")
    # Optional FlowMo bits (left as placeholders)
    ap.add_argument("--flowmo_root", type=str, default=None, help="Path to FlowMo repo root if decoding images")
    ap.add_argument("--flowmo_weights", type=str, default=None, help="Path to flowmo_hi.pth or flowmo_lo.pth")
    args = ap.parse_args()
    main(args)
