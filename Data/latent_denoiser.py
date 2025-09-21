# FlowMo Latent Denoiser – Complete, Drop‑in Pack

This pack gives you **exact files to create**, the **commands to run**, and a robust implementation to train a **Transformer denoiser over FlowMo latents**. It assumes your FlowMo repo is importable as a Python package (e.g., `pip install -e .` at the root of your project) and that your **FlowMo model + config + checkpoint** exist.

---
## 0) Target directory layout (create exactly this)

At the **root of your FlowMo repo**, create a new folder `denoiser/`:

```
<your-flowmo-repo>/
├─ flowmo/                     # your existing code (contains FlowMo, Flux, etc.)
├─ denoiser/                   # ← NEW folder we add
│  ├─ __init__.py
│  ├─ requirements.txt
│  ├─ datasets.py
│  ├─ load_flowmo.py
│  ├─ models.py
│  ├─ train.py
│  ├─ infer.py
│  └─ README.md
└─ ... (your existing files)
```

> If your repo already has a `requirements.txt`, keep this one separate under `denoiser/requirements.txt` so the denoiser stays decoupled.

---
## 1) Create files with the **exact** contents below

> **Important**: copy–paste *verbatim*. Where you must edit paths/keys, I’ll mark with `# EDIT:` comments.

### `denoiser/__init__.py`
```python
# empty file is fine; keeps this folder importable
```

### `denoiser/requirements.txt`
```text
# Minimal extra deps for the denoiser pack
omegaconf>=2.3.0
Pillow>=10.0.0
torchvision>=0.16.0
# torch must match your CUDA; install per https://pytorch.org/get-started/locally/
```

### `denoiser/datasets.py`
```python
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class ImageToLatentDataset(Dataset):
    """On-the-fly latent provider using ImageFolder.
    Returns (code[B,T,F], image_size[int]).
    """
    def __init__(self, root: str, image_size: int, flowmo_wrapper, *, antialias=True):
        super().__init__()
        self.ds = datasets.ImageFolder(
            root,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size), antialias=antialias),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0,1] -> [-1,1]
            ]),
        )
        self.flowmo = flowmo_wrapper
        self.image_size = image_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, _ = self.ds[idx]
        with torch.no_grad():
            device = next(self.flowmo.flowmo.parameters()).device
            code = self.flowmo.encode(img[None].to(device))[0].cpu()
        return code, self.image_size

class ImageNetTarToLatentDataset(Dataset):
    """Adapter for your FlowMo ImageNet tar pipeline (if present).
    Expects flowmo.data.IndexedTarDataset.
    """
    def __init__(self, imagenet_tar: str, imagenet_index: str, image_size: int, flowmo_wrapper, *, random_crop=False):
        super().__init__()
        from flowmo.data import IndexedTarDataset  # your existing loader
        self.ds = IndexedTarDataset(
            imagenet_tar=imagenet_tar,
            imagenet_index=imagenet_index,
            size=image_size,
            random_crop=random_crop,
            aug_mode="default",
        )
        self.flowmo = flowmo_wrapper
        self.image_size = image_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        # ds provides numpy image in [-1,1], HWC
        img = self.ds.preprocess_image(self.ds.index[idx])
        img = torch.from_numpy(img).permute(2,0,1)  # -> CHW
        with torch.no_grad():
            device = next(self.flowmo.flowmo.parameters()).device
            code = self.flowmo.encode(img[None].to(device))[0].cpu()
        return code, self.image_size
```

### `denoiser/load_flowmo.py`
```python
from __future__ import annotations
import torch
from omegaconf import OmegaConf

# We try a couple of import paths to find your FlowMo class.
# If FlowMo lives elsewhere, edit the import below accordingly.
try:
    from flowmo.models import FlowMo  # common placement
except Exception:
    try:
        from flowmo.model import FlowMo
    except Exception:
        from flowmo import FlowMo  # fallback if the package exposes it at top level

class FlowMoWrapper:
    """Frozen wrapper around your FlowMo model (encoder/decoder).
    Exposes:
      • encode(images[B,3,H,W] in [-1,1]) -> code[B,T,F]
      • reconstruct_from_code(code[B,T,F], image_size) -> images[B,3,H,W] in [-1,1]
    """
    def __init__(self, flowmo):
        self.flowmo = flowmo.eval()
        for p in self.flowmo.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        code, _ = self.flowmo.encode(images)
        return code

    @torch.no_grad()
    def reconstruct_from_code(self, code: torch.Tensor, image_size: int) -> torch.Tensor:
        # FlowMo.reconstruct accepts a dummy image to set HxW when code is provided
        dummy = torch.zeros(code.size(0), 3, image_size, image_size, device=code.device)
        imgs = self.flowmo.reconstruct(dummy, code=code)
        return imgs

def load_flowmo(config_path: str, ckpt_path: str, device: torch.device) -> FlowMoWrapper:
    """Load FlowMo with OmegaConf config + checkpoint.
    Prefers EMA weights if present.
    """
    cfg = OmegaConf.load(config_path)
    width = cfg.model.mup_width  # EDIT if your key differs
    model = FlowMo(width=width, config=cfg)

    state = torch.load(ckpt_path, map_location=device)
    for key in ("model_ema_state_dict", "ema", "model", "state_dict"):
        if key in state:
            missing, unexpected = model.load_state_dict(state[key], strict=False)
            print(f"[FlowMo] Loaded '{key}'. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            break
    else:
        # Fall back: assume checkpoint is a plain state_dict
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[FlowMo] Loaded raw state_dict. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return FlowMoWrapper(model)
```

### `denoiser/models.py`
```python
from __future__ import annotations
import math
import torch
import torch.nn as nn

class TimestepEmbed(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000, time_factor: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.time_factor = time_factor

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B], typically in [0,1]. Scale to match your FlowMo convention.
        t = self.time_factor * t
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=device) / float(half))
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class TransformerDenoiser(nn.Module):
    """Compact Transformer encoder over token axis (T) to denoise latents [B,T,F]."""
    def __init__(self, feature_dim: int, n_tokens: int, d_model: int = 512, n_heads: int = 8, depth: int = 6, mlp_ratio: float = 4.0, pred_target: str = "x0", time_embed_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        assert pred_target in {"x0", "eps"}
        self.pred_target = pred_target
        self.n_tokens = n_tokens
        self.feature_dim = feature_dim

        self.in_proj = nn.Linear(feature_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.t_embed = TimestepEmbed(time_embed_dim)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, d_model))

        self.out_proj = nn.Linear(d_model, feature_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, noisy_code: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, T, F = noisy_code.shape
        assert T == self.n_tokens and F == self.feature_dim, f"Expected (*,{self.n_tokens},{self.feature_dim}), got {noisy_code.shape}"
        x = self.in_proj(noisy_code)  # [B,T,d]
        x = x + self.pos_embed + self.t_proj(self.t_embed(t))[:, None, :]
        x = self.encoder(x)
        return self.out_proj(x)
```

### `denoiser/train.py`
```python
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from denoiser.load_flowmo import load_flowmo
from denoiser.datasets import ImageToLatentDataset, ImageNetTarToLatentDataset
from denoiser.models import TransformerDenoiser

@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    flowmo_config: str
    flowmo_ckpt: str

    cache_latents: bool = False  # kept for compatibility; not used with tar loader
    cache_dir: str | None = None

    batch_size: int = 64
    workers: int = 8
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    d_model: int = 512
    n_heads: int = 8
    depth: int = 6
    mlp_ratio: float = 4.0
    pred_target: str = "x0"  # or "eps"
    dropout: float = 0.0

    sigma_min: float = 0.02
    sigma_max: float = 0.4

    image_size: int = 256
    device: str = "cuda"
    amp: bool = True
    val_every: int = 1
    save_every: int = 1


def _normalize_sigma(sigmas: torch.Tensor) -> torch.Tensor:
    s_min = sigmas.min()
    s_max = sigmas.max()
    if (s_max - s_min) < 1e-8:
        return torch.zeros_like(sigmas) + 0.5
    return (sigmas - s_min) / (s_max - s_min)


def _train_step(model: TransformerDenoiser, codes: torch.Tensor, sigmas: torch.Tensor, target_mode: str):
    eps = torch.randn_like(codes)
    noisy = codes + sigmas[:, None, None] * eps
    preds = model(noisy, t=_normalize_sigma(sigmas))
    if target_mode == "x0":
        target = codes
        loss = F.mse_loss(preds, target)
    else:
        target = eps
        loss = F.mse_loss(preds, target)
    mae = (preds - target).abs().mean().item()
    return loss, {"mae": mae}


def validate_and_sample(cfg: TrainConfig, denoiser: TransformerDenoiser, flowmo, device, image_size: int, dl: DataLoader):
    denoiser.eval()
    with torch.no_grad():
        codes_cpu, _ = next(iter(dl))
        codes = codes_cpu.to(device)
        B, T, Fdim = codes.shape
        sigma = torch.full((B,), (cfg.sigma_min + cfg.sigma_max) * 0.5, device=device)
        noisy = codes + sigma[:, None, None] * torch.randn_like(codes)
        preds = denoiser(noisy, t=_normalize_sigma(sigma))
        recon = flowmo.reconstruct_from_code(preds, image_size).clamp(-1, 1)
        grid = make_grid((recon + 1) * 0.5, nrow=min(4, B))
        out_path = Path(cfg.out_dir) / "samples" / f"val_epoch_sample.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(grid, out_path)
        print(f"[val] Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Train a Transformer denoiser over FlowMo latents")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--flowmo_config", type=str, required=True)
    p.add_argument("--flowmo_ckpt", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--pred_target", choices=["x0", "eps"], default="x0")
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--sigma_min", type=float, default=0.02)
    p.add_argument("--sigma_max", type=float, default=0.4)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=1)

    args = p.parse_args()
    cfg = TrainConfig(**vars(args))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    (Path(cfg.out_dir) / "ckpts").mkdir(parents=True, exist_ok=True)
    (Path(cfg.out_dir) / "samples").mkdir(parents=True, exist_ok=True)

    flowmo = load_flowmo(cfg.flowmo_config, cfg.flowmo_ckpt, device)

    # Dataset selection: tar pipeline or ImageFolder
    if cfg.data_root.endswith(".tar"):
        # Try to infer the index path next to the tar; edit if yours differs.
        idx_guess = cfg.data_root.replace(".tar", "_index_overall.json").replace("train", "train")
        ds = ImageNetTarToLatentDataset(cfg.data_root, idx_guess, cfg.image_size, flowmo, random_crop=False)
    else:
        ds = ImageToLatentDataset(cfg.data_root, cfg.image_size, flowmo)

    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=True)

    # Lazy init of denoiser
    codes0, imgsize0 = next(iter(dl))
    T, Fdim = codes0.shape[1], codes0.shape[2]
    denoiser = TransformerDenoiser(feature_dim=Fdim, n_tokens=T, d_model=cfg.d_model, n_heads=cfg.n_heads, depth=cfg.depth, mlp_ratio=cfg.mlp_ratio, pred_target=cfg.pred_target, dropout=cfg.dropout).to(device)

    opt = torch.optim.AdamW(denoiser.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        denoiser.train()
        ema_loss = 0.0
        t0 = time.time()
        for codes_cpu, _ in dl:
            codes = codes_cpu.to(device)
            sigmas = torch.rand(codes.size(0), device=device) * (cfg.sigma_max - cfg.sigma_min) + cfg.sigma_min
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp, dtype=torch.bfloat16 if cfg.amp else None):
                loss, logs = _train_step(denoiser, codes, sigmas, cfg.pred_target)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            ema_loss = 0.9 * ema_loss + 0.1 * loss.item() if global_step > 0 else loss.item()
            if global_step % 50 == 0:
                print(f"ep {epoch:03d} | step {global_step:06d} | loss {loss.item():.4f} | ema {ema_loss:.4f} | mae {logs['mae']:.4f}")
            global_step += 1

        print(f"Epoch {epoch} done in {(time.time()-t0)/60:.2f} min | EMA {ema_loss:.4f}")

        if (epoch % cfg.val_every) == 0:
            validate_and_sample(cfg, denoiser, flowmo, device, cfg.image_size, dl)
        if (epoch % cfg.save_every) == 0:
            ckpt = Path(cfg.out_dir) / "ckpts" / f"denoiser_e{epoch:03d}.pt"
            torch.save({"model": denoiser.state_dict(), "cfg": asdict(cfg), "epoch": epoch, "step": global_step}, ckpt)
            print(f"[ckpt] Saved: {ckpt}")

if __name__ == "__main__":
    main()
```

### `denoiser/infer.py`
```python
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from denoiser.load_flowmo import load_flowmo
from denoiser.models import TransformerDenoiser

@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Denoise precomputed FlowMo latents and decode")
    p.add_argument("--flowmo_config", type=str, required=True)
    p.add_argument("--flowmo_ckpt", type=str, required=True)
    p.add_argument("--denoiser_ckpt", type=str, required=True)
    p.add_argument("--latents_path", type=str, required=True, help=".pt containing {'code': [B,T,F], 'image_size': int}")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    flowmo = load_flowmo(args.flowmo_config, args.flowmo_ckpt, device)

    blob = torch.load(args.latents_path, map_location=device)
    codes = blob["code"].to(device)
    image_size = int(blob["image_size"]) if "image_size" in blob else 256

    # Recreate denoiser from shape
    B, T, Fdim = codes.shape
    den = TransformerDenoiser(feature_dim=Fdim, n_tokens=T)
    state = torch.load(args.denoiser_ckpt, map_location=device)
    den.load_state_dict(state["model"])
    den.to(device).eval()

    # Small fixed sigma used only to define the conditioning input
    sigma = torch.full((B,), 0.1, device=device)
    preds = den(codes, t=(sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-8))

    recon = flowmo.reconstruct_from_code(preds, image_size).clamp(-1, 1)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(recon):
        save_image((img + 1) * 0.5, out_dir / f"recon_{i:04d}.png")
        print("saved", out_dir / f"recon_{i:04d}.png")

if __name__ == "__main__":
    main()
```

### `denoiser/README.md`
```markdown
# FlowMo Latent Denoiser (Pack)

Train a small Transformer to denoise FlowMo latents, then decode with FlowMo.

## 1) Install

```bash
# From the root of your FlowMo repo
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r denoiser/requirements.txt
# Install your FlowMo package itself (so `import flowmo` works):
pip install -e .
```

> Install PyTorch that matches your CUDA first if you haven’t already.

## 2) Train

**ImageFolder** (directory of images):
```bash
python -m denoiser.train \
  --data_root /path/to/images_dir \
  --flowmo_config flowmo/configs/base.yaml \
  --flowmo_ckpt /path/to/flowmo.ckpt \
  --out_dir runs/latent_denoiser \
  --batch_size 64 --epochs 30 --amp \
  --sigma_min 0.02 --sigma_max 0.4 \
  --pred_target x0 \
  --image_size 256
```

**ImageNet tar + index** (your loader):
```bash
python -m denoiser.train \
  --data_root /data/ILSVRC2012_img_train.tar \
  --flowmo_config flowmo/configs/base.yaml \
  --flowmo_ckpt /path/to/flowmo.ckpt \
  --out_dir runs/latent_denoiser \
  --batch_size 64 --epochs 30 --amp \
  --sigma_min 0.02 --sigma_max 0.4 \
  --pred_target x0 \
  --image_size 256
```

Artifacts:
- Checkpoints: `runs/latent_denoiser/ckpts/denoiser_eXXX.pt`
- Validation grid: `runs/latent_denoiser/samples/val_epoch_sample.png`

## 3) Inference on saved latents

Save a `.pt` with `{ "code": [B,T,F], "image_size": 256 }`, then:
```bash
python -m denoiser.infer \
  --flowmo_config flowmo/configs/base.yaml \
  --flowmo_ckpt /path/to/flowmo.ckpt \
  --denoiser_ckpt runs/latent_denoiser/ckpts/denoiser_e030.pt \
  --latents_path /path/to/latents.pt \
  --out_dir outputs/recons
```

## 4) Notes & Tips
- With **LFQ** quantization (your default), `F = context_dim` (e.g., 18). With **KL**, `F = 2 * context_dim` *before* sampling. The code lazily infers (T,F).
- If outputs look too smooth, lower `--sigma_max` or try `--pred_target eps`.
- If unstable, try disabling `--amp` briefly, then re-enable once stable.
- The denoiser conditions on a scalar noise level `σ` (mapped to [0,1]) via sinusoidal embedding.
- For long `T`, reduce `depth` or `d_model` to fit memory.
```

---
## 2) One-time edits you may need

- **Import path** for `FlowMo` in `denoiser/load_flowmo.py`.
  - If your class lives at `flowmo/flowmo.py` (the single file you pasted), change to:
    ```python
    from flowmo.flowmo import FlowMo
    ```
- **Config keys**: If your config doesn’t have `model.mup_width`, set a constant width that matches how you trained, or compute it from your existing values, e.g. `width = 4` for `dit_b_4` equivalent.

---
## 3) Why this will “just work” with your repo

- We do **not** touch your training code; we only import your model in **eval** mode and use its **encoder**/**decoder**.
- The denoiser initializes itself after seeing the first `[B,T,F]` so there’s no mismatch.
- Validation uses your **existing** `reconstruct` flow (sampler, schedules, CFG handling) for faithful qualitative checks.

---
## 4) Common pitfalls & fixes

- **ModuleNotFoundError: flowmo** → run `pip install -e .` at your repo root so `import flowmo` is resolvable.
- **Checkpoint key mismatch** → we try EMA → model → state_dict. If still missing, print keys in your ckpt and adjust the loader.
- **OOM** → lower `--batch_size`, `--d_model`, `--depth`, or image size; enable `--amp`.
- **Blurry** reconstructions → reduce `--sigma_max`, try `--pred_target eps`, or add `(1/σ²)` weighting in the loss (can be added in `train.py`).

---
## 5) Next steps (optional improvements)

- Replace additive time-bias with **FiLM (scale/shift)** per block (mirrors your Modulation blocks); easy to slot into `models.py`.
- Add an LPIPS-based validation metric (image-space) while keeping training purely latent MSE for speed.
- Per-token noise schedules: draw σ per token and pass a pooled σ to the embedding (or enrich the model to consume a `[B,T]` time tensor).
