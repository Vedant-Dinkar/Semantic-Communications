# regen_awgn_single_var.py
# One-CSV AWGN regeneration with per-row sigma (and optional per-position jitter).
# Input CSV: [first_col, orig_0..orig_{L-1}, corr_0..corr_{L-1}] (with or without header)
# Output CSV: same schema, same row count, first_col and original half preserved,
#             corrupted half replaced using AWGN with per-row sigma.

import argparse, os
import numpy as np
import pandas as pd

def drop_unnamed_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    return df.drop(columns=cols, errors="ignore") if cols else df

def bytes_to_float_pm1(u8: np.ndarray) -> np.ndarray:
    return (u8.astype(np.float32) / 255.0) * 2.0 - 1.0

def float_pm1_to_bytes(x: np.ndarray) -> np.ndarray:
    y = np.clip((x + 1.0) * 0.5, 0.0, 1.0) * 255.0
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)

def infer_L_from_ncols(ncols: int) -> int:
    if (ncols - 1) % 2 != 0:
        raise ValueError(f"Bad column count {ncols}; expected 1 + 2L.")
    return (ncols - 1) // 2

def detect_has_header(in_csv: str) -> tuple[bool, list[str] | None]:
    # Peek raw first line (header=None) to avoid pandas inference traps.
    peek = pd.read_csv(in_csv, header=None, nrows=1, dtype=str, na_filter=False)
    peek = drop_unnamed_cols(peek)
    row0 = peek.iloc[0].tolist()
    has_header = False
    for tok in row0:
        try:
            float(tok)
        except Exception:
            has_header = True
            break
    colnames = None
    if has_header:
        head_df = pd.read_csv(in_csv, header=0, nrows=0)
        head_df = drop_unnamed_cols(head_df)
        colnames = list(head_df.columns)
    return has_header, colnames

def sample_sigma_per_row(rng: np.random.Generator, mode: str,
                         sigma_range: tuple[float,float] | None,
                         sigma_set: list[float] | None) -> float:
    if mode == "uniform":
        lo, hi = sigma_range
        return float(rng.uniform(lo, hi))
    elif mode == "set":
        return float(rng.choice(np.array(sigma_set, dtype=np.float32)))
    else:
        raise ValueError("sigma_mode must be 'uniform' or 'set'")

def make_positional_jitter(T: int, rng: np.random.Generator,
                           jitter_amp: float, jitter_freq: float) -> np.ndarray:
    """
    Returns multiplicative jitter vector of shape [T], mean ~1.0.
    jitter_amp in [0,1): amplitude of modulation (e.g., 0.2 -> ±20%).
    jitter_freq: cycles across the sequence (e.g., 1.0 means one cycle over T).
    """
    if jitter_amp <= 0:
        return np.ones((T,), dtype=np.float32)
    i = np.arange(T, dtype=np.float32)
    phase = rng.uniform(0, 2*np.pi)
    base = 1.0 + jitter_amp * np.sin(2*np.pi*jitter_freq * (i / max(1,T-1)) + phase)
    return base.astype(np.float32)

def awgn_bytes(orig_bytes_row: np.ndarray, sigma: float,
               rng: np.random.Generator,
               jitter_amp: float = 0.0, jitter_freq: float = 1.0) -> np.ndarray:
    """
    Add AWGN in [-1,1] space with per-row sigma and optional per-position jitter.
    """
    x = bytes_to_float_pm1(orig_bytes_row)                 # [-1,1], float32
    if jitter_amp > 0:
        jitter = make_positional_jitter(x.shape[0], rng, jitter_amp, jitter_freq)  # [T]
        local_sigma = sigma * jitter
    else:
        local_sigma = sigma
    n = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    y = np.clip(x + n * local_sigma, -1.0, 1.0)
    return float_pm1_to_bytes(y)                           # uint8 [0..255]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",  required=True)
    ap.add_argument("--out_csv", required=True)
    # choose one of the sigma modes:
    ap.add_argument("--sigma_mode", choices=["uniform", "set"], default="uniform")
    ap.add_argument("--sigma_lo", type=float, default=0.03, help="uniform low (if sigma_mode=uniform)")
    ap.add_argument("--sigma_hi", type=float, default=0.15, help="uniform high (if sigma_mode=uniform)")
    ap.add_argument("--sigmas",   type=float, nargs="*", default=None, help="list of sigmas (if sigma_mode=set)")
    # optional per-position heteroscedasticity
    ap.add_argument("--jitter_amp",  type=float, default=0.0, help="0..<1, e.g., 0.2 for ±20%")
    ap.add_argument("--jitter_freq", type=float, default=1.0, help="cycles across sequence (e.g., 1.0 means one cycle)")
    ap.add_argument("--chunksize",   type=int, default=20000)
    ap.add_argument("--seed",        type=int, default=1234)
    args = ap.parse_args()

    assert os.path.isfile(args.in_csv), f"Missing input: {args.in_csv}"
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # header handling
    has_header, colnames = detect_has_header(args.in_csv)
    if has_header:
        it = pd.read_csv(args.in_csv, header=0, dtype=str, na_filter=False, chunksize=args.chunksize)
    else:
        it = pd.read_csv(args.in_csv, header=None, dtype=str, na_filter=False, chunksize=args.chunksize)

    # configure sigma sampler
    if args.sigma_mode == "uniform":
        sigma_range = (float(args.sigma_lo), float(args.sigma_hi))
        sigma_set = None
    else:
        assert args.sigmas and len(args.sigmas) > 0, "Provide --sigmas when sigma_mode=set"
        sigma_range = None
        sigma_set = [float(s) for s in args.sigmas]

    wrote_header = False
    total_rows_out = 0

    with open(args.out_csv, "w", newline="") as fout:
        for chunk in it:
            chunk = drop_unnamed_cols(chunk)

            # Drop any stray repeated header lines inside the file
            mfirst = pd.to_numeric(chunk.iloc[:, 0], errors="coerce").notna()
            if not mfirst.all():
                chunk = chunk[mfirst]
            if chunk.empty:
                continue

            # Convert to numeric
            chunk = chunk.apply(pd.to_numeric, errors="raise")
            vals = chunk.to_numpy()               # [N, 1+2L]
            ncols = vals.shape[1]
            L = infer_L_from_ncols(ncols)

            first_col = vals[:, :1]                         # preserved verbatim
            orig      = vals[:, 1:1+L].astype(np.uint8)     # preserved verbatim

            # Per-row sigma sampling
            N = orig.shape[0]
            sigmas_row = np.empty((N,), dtype=np.float32)
            for i in range(N):
                sigmas_row[i] = sample_sigma_per_row(rng, args.sigma_mode, sigma_range, sigma_set)

            # Regenerate corrupted half
            corr = np.empty_like(orig, dtype=np.uint8)
            for i in range(N):
                corr[i] = awgn_bytes(
                    orig[i], sigma=float(sigmas_row[i]), rng=rng,
                    jitter_amp=float(args.jitter_amp), jitter_freq=float(args.jitter_freq)
                )

            out_block = np.concatenate([first_col, orig, corr], axis=1)

            if has_header and not wrote_header:
                fout.write(",".join(colnames) + "\n")
                wrote_header = True

            for r in out_block:
                fout.write(",".join(map(str, r.tolist())) + "\n")
            total_rows_out += out_block.shape[0]

    print(f"[Done] wrote: {args.out_csv}")
    print(f"  header: {'preserved' if has_header else 'none'}")
    print(f"  rows: {total_rows_out}")
    print(f"  sigma_mode: {args.sigma_mode} | "
          f"{('[%.3f, %.3f]' % (args.sigma_lo, args.sigma_hi)) if args.sigma_mode=='uniform' else args.sigmas} | "
          f"jitter_amp={args.jitter_amp} jitter_freq={args.jitter_freq}")

if __name__ == "__main__":
    main()
