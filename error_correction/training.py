# csv_byte_denoiser.py
# SOTA-style Conformer-U-Net for denoising byte streams stored as CSV:
# Row format: [BER, orig_0..orig_{L-1}, corr_0..corr_{L-1}]
# We IGNORE the BER column entirely.

import os, math, glob, gc, argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as cp


# =========================
# Utils
# =========================

def set_seed(sd=42):
    import random
    random.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)

def bytes_to_float(u8: torch.Tensor, mode="pm1"):
    # bytes [0..255] -> float in [-1,1] (default) or [0,1]
    if mode == "pm1":
        return (u8.float() / 255.0) * 2.0 - 1.0
    elif mode == "01":
        return u8.float() / 255.0
    raise ValueError(mode)

def float_to_u8(x: torch.Tensor, mode="pm1"):
    if mode == "pm1":
        y = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0) * 255.0
    else:
        y = torch.clamp(x, 0.0, 1.0) * 255.0
    return y.round().long().clamp(0, 255)

def symbol_accuracy(target: torch.Tensor, pred: torch.Tensor) -> float:
    return float((target == pred).float().mean().item())

def bit_accuracy(target: torch.Tensor, pred: torch.Tensor) -> float:
    # per-bit compare for 8-bit symbols
    mask = (1 << torch.arange(8, device=target.device)).view(1,1,8)
    tb = ((target.unsqueeze(-1) & mask) != 0)
    pb = ((pred.unsqueeze(-1) & mask) != 0)
    return float((tb == pb).float().mean().item())


# =========================
# Dataset (robust CSV)
# =========================

class CsvBytePairs(Dataset):
    """
    Accepts CSVs whose rows are:
      [BER, orig_0..orig_{L-1}, corr_0..corr_{L-1}]
    - Ignores BER.
    - Auto-detects header (first row non-numeric).
    - Drops stray "Unnamed: *" columns if present.
    Returns:
      noisy_f: [T,1] float (scaled from corrupted bytes)
      clean  : [T]   long  (0..255)
    """
    def __init__(self, csv_paths: List[str], scale_mode="pm1", enforce_len: int = 0):
        assert len(csv_paths) > 0, "No CSV paths provided"
        self.scale_mode = scale_mode
        frames = []
        for p in csv_paths:
            # initial read (string) to detect header
            df_raw = pd.read_csv(p, dtype=str, na_filter=False)
            # drop unnamed columns
            drop_cols = [c for c in df_raw.columns if str(c).startswith("Unnamed")]
            if drop_cols:
                df_raw = df_raw.drop(columns=drop_cols, errors="ignore")

            def _is_numeric_series(s):
                try:
                    pd.to_numeric(s, errors="raise")
                    return True
                except Exception:
                    return False

            has_header = not _is_numeric_series(df_raw.iloc[0, :])

            if has_header:
                df = pd.read_csv(p, header=0, na_filter=False, dtype=str)
                if drop_cols:
                    df = df.drop(columns=drop_cols, errors="ignore")
            else:
                df = df_raw

            # convert to numeric
            df = df.apply(pd.to_numeric, errors="raise")
            if df.shape[1] < 3:
                raise ValueError(f"{p}: expected >= 3 columns, got {df.shape[1]}")
            frames.append(df)

        big = pd.concat(frames, axis=0, ignore_index=True)
        arr = big.to_numpy()  # [N, 1+2L]
        ncols = arr.shape[1]
        if (ncols - 1) % 2 != 0:
            raise ValueError(f"Bad column count {ncols}; expected 1 + 2L.")

        self.L = (ncols - 1) // 2
        self.arr = arr.astype(np.int16)

        # optional enforce sequence length (truncate/pad)
        self.enforce_len = enforce_len

    def __len__(self): return self.arr.shape[0]

    def __getitem__(self, i):
        row = self.arr[i]                # [1 + 2L]
        payload = row[1:]                # drop BER
        orig = payload[:self.L].astype(np.int64)
        corr = payload[self.L:].astype(np.int64)

        clean = torch.from_numpy(orig).long()        # [T]
        noisy = torch.from_numpy(corr).long()        # [T]

        if self.enforce_len > 0 and self.enforce_len != clean.size(0):
            T0 = clean.size(0)
            T = self.enforce_len
            if T0 >= T:
                clean = clean[:T]
                noisy = noisy[:T]
            else:
                padT = T - T0
                clean = torch.cat([clean, clean.new_zeros(padT)], dim=0)
                noisy = torch.cat([noisy, noisy.new_zeros(padT)], dim=0)

        noisy_f = bytes_to_float(noisy, self.scale_mode).unsqueeze(-1)  # [T,1]
        return noisy_f, clean


# =========================
# Model: Conformer-U-Net
# =========================

class SinusoidalPosEnc(nn.Module):
    def __init__(self, dim: int, max_len: int = 1<<18):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, L: int): return self.pe[:L]

class GLU(nn.Module):
    def __init__(self, d): super().__init__(); self.proj = nn.Linear(d, 2*d)
    def forward(self, x): a,b = self.proj(x).chunk(2, dim=-1); return a * torch.sigmoid(b)

class DepthwiseConv1d(nn.Module):
    def __init__(self, d, k=15, dil=1):
        super().__init__()
        pad = (k//2)*dil
        self.dw = nn.Conv1d(d, d, k, groups=d, padding=pad, dilation=dil)
        self.pw = nn.Conv1d(d, d, 1); self.act = nn.SiLU()
    def forward(self, x):  # [B,T,d]
        y = x.transpose(1,2)
        y = self.dw(y); y = self.act(y); y = self.pw(y)
        return y.transpose(1,2)

class SqueezeExcite(nn.Module):
    def __init__(self, d, r=0.25):
        super().__init__()
        h = max(8, int(d*r))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d, h, 1), nn.SiLU(),
            nn.Conv1d(h, d, 1), nn.Sigmoid()
        )
    def forward(self, x):  # [B,T,d]
        y = x.transpose(1,2); y = self.net(y)
        return (x.transpose(1,2) * y).transpose(1,2)

class ConformerBlock(nn.Module):
    def __init__(self, d, nhead=8, ff_mult=4, drop=0.05, k=15):
        super().__init__()
        ff = d*ff_mult
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,ff), nn.SiLU(), nn.Dropout(drop), nn.Linear(ff,d))
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.conv_norm = nn.LayerNorm(d)
        self.glu = GLU(d); self.dw = DepthwiseConv1d(d,k); self.se = SqueezeExcite(d); self.conv_out = nn.Linear(d,d)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,ff), nn.SiLU(), nn.Dropout(drop), nn.Linear(ff,d))
        self.dropout = nn.Dropout(drop)
    def forward(self, x, kpm=None):
        x = x + 0.5*self.dropout(self.ff1(x))
        h = self.attn_norm(x); a,_ = self.attn(h,h,h, key_padding_mask=kpm, need_weights=False); x = x + self.dropout(a)
        h = self.conv_norm(x); h = self.glu(h); h = self.dw(h); h = self.se(h); h = self.conv_out(h); x = x + self.dropout(h)
        x = x + 0.5*self.dropout(self.ff2(x))
        return x

class Down1D(nn.Module):
    def __init__(self, di, do, s=2):
        super().__init__()
        self.conv = nn.Conv1d(di, do, 4, stride=s, padding=1)
        self.norm = nn.GroupNorm(8, do); self.act = nn.SiLU()
    def forward(self, x): y = x.transpose(1,2); y = self.conv(y); y = self.norm(y); y = self.act(y); return y.transpose(1,2)

class Up1D(nn.Module):
    def __init__(self, di, do, s=2):
        super().__init__()
        self.tconv = nn.ConvTranspose1d(di, do, 4, stride=s, padding=1)
        self.norm = nn.GroupNorm(8, do); self.act = nn.SiLU()
    def forward(self, x): y = x.transpose(1,2); y = self.tconv(y); y = self.norm(y); y = self.act(y); return y.transpose(1,2)

@dataclass
class ModelCfg:
    d_model: int = 512
    nhead: int = 8
    ff_mult: int = 4
    dropout: float = 0.05
    conv_kernel: int = 15
    depths_down: Tuple[int,int] = (2,2)
    depths_mid: int = 4
    depths_up: Tuple[int,int] = (2,2)
    use_posenc: bool = True

def _ckpt_block(block: nn.Module, x: torch.Tensor, use_ckpt: bool):
    if use_ckpt:
        return cp.checkpoint(lambda t: block(t), x, use_reentrant=False)
    return block(x)

class ByteDenoiser(nn.Module):
    """
    Input : noisy float [B,T,1]  (scaled corrupted bytes)
    Output: logits [B,T,256]     (classify clean byte per position)
    """
    def __init__(self, cfg: ModelCfg, use_ckpt: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_ckpt = use_ckpt
        d = cfg.d_model

        self.inp = nn.Linear(1, d)
        self.pos = SinusoidalPosEnc(d) if cfg.use_posenc else None

        # Down
        self.enc1 = nn.ModuleList([ConformerBlock(d, cfg.nhead, cfg.ff_mult, cfg.dropout, cfg.conv_kernel) for _ in range(cfg.depths_down[0])])
        self.down1 = Down1D(d, d*2); d2 = d*2
        self.enc2 = nn.ModuleList([ConformerBlock(d2, cfg.nhead, cfg.ff_mult, cfg.dropout, cfg.conv_kernel) for _ in range(cfg.depths_down[1])])
        self.down2 = Down1D(d2, d2*2); d3 = d2*2

        # Mid
        self.mid = nn.ModuleList([ConformerBlock(d3, cfg.nhead, cfg.ff_mult, cfg.dropout, cfg.conv_kernel) for _ in range(cfg.depths_mid)])

        # Up
        self.up2 = Up1D(d3, d2)
        self.dec2 = nn.ModuleList([ConformerBlock(d2, cfg.nhead, cfg.ff_mult, cfg.dropout, cfg.conv_kernel) for _ in range(cfg.depths_up[0])])
        self.up1 = Up1D(d2, d)
        self.dec1 = nn.ModuleList([ConformerBlock(d,  cfg.nhead, cfg.ff_mult, cfg.dropout, cfg.conv_kernel) for _ in range(cfg.depths_up[1])])

        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 256)

    def forward(self, noisy: torch.Tensor):
        B, T, _ = noisy.shape
        x = self.inp(noisy)
        if self.pos is not None:
            x = x + self.pos(T).to(x.device, x.dtype).unsqueeze(0)

        for blk in self.enc1:
            x = _ckpt_block(blk, x, self.use_ckpt)
        s1 = x
        x = self.down1(x)

        for blk in self.enc2:
            x = _ckpt_block(blk, x, self.use_ckpt)
        s2 = x
        x = self.down2(x)

        for blk in self.mid:
            x = _ckpt_block(blk, x, self.use_ckpt)

        x = self.up2(x); x = x + s2
        for blk in self.dec2:
            x = _ckpt_block(blk, x, self.use_ckpt)

        x = self.up1(x); x = x + s1
        for blk in self.dec1:
            x = _ckpt_block(blk, x, self.use_ckpt)

        x = self.norm(x)
        return self.out(x)   # [B,T,256]


# =========================
# Training / Eval
# =========================

@dataclass
class TrainCfg:
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    total_steps: int = 20000
    warmup_steps: int = 1000
    log_every: int = 50
    val_every: int = 500
    grad_clip: float = 1.0
    grad_accum: int = 2
    ema_decay: float = 0.999
    amp: bool = True
    label_smoothing: float = 0.0

def cosine_warmup(opt, total_steps, warmup_steps):
    def fn(step):
        if step < warmup_steps:
            return (step+1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = min(1.0, max(0.0, t))
        return 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)

def label_smoothing_ce(logits, target, eps=0.0):
    # logits: [B,T,256], target: [B,T]
    if eps == 0.0:
        return nn.functional.cross_entropy(logits.transpose(1,2), target, reduction="mean")
    B,T,C = logits.shape
    logp = nn.functional.log_softmax(logits, dim=-1)
    with torch.no_grad():
        hard = torch.zeros_like(logp).scatter_(-1, target.unsqueeze(-1), 1.0)
        soft = (1.0 - eps) * hard + eps / C
    return -(soft * logp).sum(dim=-1).mean()

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> Tuple[float,float]:
    model.eval()
    sym_accs, bit_accs = [], []
    with torch.no_grad(), autocast(enabled=(device=="cuda")):
        for noisy_f, clean in loader:
            noisy_f = noisy_f.to(device); clean = clean.to(device)
            logits = model(noisy_f)                 # [B,T,256]
            pred = logits.argmax(dim=-1).long()     # [B,T]
            sym_accs.append(symbol_accuracy(clean, pred))
            bit_accs.append(bit_accuracy(clean, pred))
    model.train()
    return float(np.mean(sym_accs)), float(np.mean(bit_accs))


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--csv_glob", type=str, default="*.csv", help="Glob for CSVs")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seq_len", type=int, default=0, help="0=use CSV inferred length")

    # training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--total_steps", type=int, default=20000)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--micro_splits", type=int, default=0,
                    help=">0 to split each batch into this many micro-chunks to reduce peak memory")

    # model size
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--conv_kernel", type=int, default=15)
    ap.add_argument("--depths_down1", type=int, default=2)
    ap.add_argument("--depths_down2", type=int, default=2)
    ap.add_argument("--depths_mid", type=int, default=4)
    ap.add_argument("--depths_up1", type=int, default=2)
    ap.add_argument("--depths_up2", type=int, default=2)
    ap.add_argument("--no_posenc", action="store_true")
    ap.add_argument("--ckpt_activations", action="store_true")

    # modes
    ap.add_argument("--mode", type=str, default="train", choices=["train","eval","infer_one"])
    ap.add_argument("--ckpt_out", type=str, default="byte_denoiser_best_ema.pt")
    ap.add_argument("--in_csv", type=str, default="")
    ap.add_argument("--row_index", type=int, default=0)

    args = ap.parse_args()

    # perf toggles (harmless on CPU)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(42)
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available() and torch.version.cuda) else "cpu"
    print(f"[Device] {device}")

    csvs = sorted(glob.glob(args.csv_glob))
    assert csvs, f"No CSVs match: {args.csv_glob}"

    # dataset
    full = CsvBytePairs(csvs, scale_mode="pm1", enforce_len=args.seq_len)
    T = full.L if args.seq_len == 0 else args.seq_len
    print(f"[Data] rows={len(full)}  T={T}  (inferred L={full.L})")

    n_total = len(full)
    n_val = max(1, int(n_total * args.val_frac))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(123))

    def collate(batch):
        noisy_f, clean = zip(*batch)
        return torch.stack(noisy_f, 0), torch.stack(clean, 0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate)

    # model
    mcfg = ModelCfg(
        d_model=args.d_model, nhead=args.nhead, ff_mult=args.ff_mult,
        dropout=0.05, conv_kernel=args.conv_kernel,
        depths_down=(args.depths_down1, args.depths_down2),
        depths_mid=args.depths_mid, depths_up=(args.depths_up1, args.depths_up2),
        use_posenc=(not args.no_posenc)
    )
    model = ByteDenoiser(mcfg, use_ckpt=args.ckpt_activations).to(device)

    if args.mode == "eval":
        ckpt = torch.load(args.ckpt_out, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        sym, bit = evaluate(model.eval(), val_loader, device)
        print(f"[Eval] SymbolAcc={sym:.4f}  BitAcc={bit:.4f}")
        return

    if args.mode == "infer_one":
        ckpt = torch.load(args.ckpt_out, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        assert os.path.isfile(args.in_csv), "--in_csv required"
        df = pd.read_csv(args.in_csv, dtype=str, na_filter=False)
        # drop unnamed
        drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if drop_cols: df = df.drop(columns=drop_cols, errors="ignore")
        # detect header
        try:
            pd.to_numeric(df.iloc[0, :], errors="raise")
            has_header = False
        except Exception:
            has_header = True
        if has_header:
            df = pd.read_csv(args.in_csv, header=0, na_filter=False, dtype=str)
            if drop_cols: df = df.drop(columns=drop_cols, errors="ignore")
        df = df.apply(pd.to_numeric, errors="raise")
        row = df.iloc[args.row_index].to_numpy()
        ncols = row.shape[0]; L = (ncols - 1)//2
        corr = torch.tensor(row[1+L:], dtype=torch.long)
        noisy_f = bytes_to_float(corr, "pm1").unsqueeze(0).unsqueeze(-1).to(device)  # [1,T,1]
        with torch.no_grad(), autocast(enabled=(device=="cuda" and args.amp)):
            logits = model(noisy_f); pred = logits.argmax(dim=-1)[0].cpu().tolist()
        print("pred[:64]:", pred[:64])
        return

    # training setup
    tcfg = TrainCfg(
        batch_size=args.batch_size, lr=args.lr, total_steps=args.total_steps,
        grad_accum=args.grad_accum, amp=args.amp, label_smoothing=args.label_smoothing
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay, betas=(0.9,0.95))
    sched = cosine_warmup(opt, total_steps=tcfg.total_steps, warmup_steps=tcfg.warmup_steps)
    scaler = GradScaler(enabled=tcfg.amp)

    # EMA (lightweight dict)
    ema = {"shadow": {k: v.detach().clone() for k, v in model.state_dict().items()}, "decay": 0.999}
    def ema_update(m):
        with torch.no_grad():
            for k, v in m.state_dict().items():
                ema["shadow"][k].mul_(ema["decay"]).add_(v.detach(), alpha=1 - ema["decay"])
    def ema_copy_to(m):
        m.load_state_dict(ema["shadow"], strict=False)

    step = 0
    best_sym = -1.0
    opt.zero_grad(set_to_none=True)
    model.train()

    while step < tcfg.total_steps:
        for noisy_f, clean in train_loader:
            noisy_f = noisy_f.to(device); clean = clean.to(device)

            # Optional micro-chunking to reduce peak memory
            if args.micro_splits and args.micro_splits > 1:
                total_loss = 0.0
                splits_nf = noisy_f.split(max(1, noisy_f.size(0) // args.micro_splits), dim=0)
                splits_cl = clean.split(max(1, clean.size(0) // args.micro_splits), dim=0)
                for nf_chunk, cl_chunk in zip(splits_nf, splits_cl):
                    with autocast(enabled=tcfg.amp):
                        logits = model(nf_chunk)
                        loss = label_smoothing_ce(logits, cl_chunk, eps=tcfg.label_smoothing) / max(1, tcfg.grad_accum)
                    total_loss += float(loss.detach().cpu())
                    scaler.scale(loss).backward()
            else:
                with autocast(enabled=tcfg.amp):
                    logits = model(noisy_f)
                    loss = label_smoothing_ce(logits, clean, eps=tcfg.label_smoothing) / max(1, tcfg.grad_accum)
                scaler.scale(loss).backward()

            if (step + 1) % tcfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                ema_update(model)
                sched.step()

            step += 1
            if step % tcfg.log_every == 0:
                print(f"[Step {step:06d}] lr={opt.param_groups[0]['lr']:.2e}")

            if step % tcfg.val_every == 0:
                ema_model = ByteDenoiser(mcfg, use_ckpt=False).to(device).eval()
                ema_copy_to(ema_model)
                sym, bit = evaluate(ema_model, val_loader, device)
                print(f"[Val @ {step}] EMA SymbolAcc={sym:.4f}  BitAcc={bit:.4f}  (bestSym={best_sym:.4f})")
                if sym > best_sym:
                    best_sym = sym
                    torch.save({"model": ema_model.state_dict(), "mcfg": mcfg.__dict__}, args.ckpt_out)
                    print(f"[Checkpoint] Saved {args.ckpt_out}")
                del ema_model; gc.collect()
                if device == "cuda": torch.cuda.empty_cache()

            if step >= tcfg.total_steps:
                break

    print(f"[Train] Done. Best EMA SymbolAcc={best_sym:.4f}")
    ema_model = ByteDenoiser(mcfg, use_ckpt=False).to(device).eval()
    ema_copy_to(ema_model)
    sym, bit = evaluate(ema_model, val_loader, device)
    print(f"[Final Eval] EMA SymbolAcc={sym:.4f}  BitAcc={bit:.4f}")
    torch.save({"model": ema_model.state_dict(), "mcfg": mcfg.__dict__}, args.ckpt_out)
    print(f"[Saved] {args.ckpt_out}")


if __name__ == "__main__":
    main()
