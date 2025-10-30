# Imports and basic setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Dataset path
CSV_PATH = './binary_code_awgn_dataset.csv'

print('Notebook imports ready')




# Data loading and Dataset class

df = pd.read_csv(CSV_PATH)
print('Loaded CSV shape:', df.shape)

# identify columns
orig_cols = [c for c in df.columns if c.startswith('Original_')]
cor_cols = [c for c in df.columns if c.startswith('Corrupted_')]
print('Found', len(orig_cols), 'original cols and', len(cor_cols), 'corrupted cols')

# We'll use the corrupted columns as input and original columns as target. The values are -1 or 1.

class AWGNDataset(Dataset):
    def __init__(self, df, corrupted_cols, original_cols):
        self.cor = df[corrupted_cols].values.astype(np.float32)
        self.orig = df[original_cols].values.astype(np.float32)
        # Convert from -1/1 to 0/1 for binary BCE training if desired, but we'll keep -1/1 and use MSE
        # We'll reshape to (N, seq_len, input_dim_per_token) if we want tokens; here we treat whole vector as one token
        # So input_dim = number of corrupted columns
    def __len__(self):
        return len(self.cor)
    def __getitem__(self, idx):
        x = self.cor[idx]
        y = self.orig[idx]
        # Return as tensors
        return torch.from_numpy(x), torch.from_numpy(y)

# Quick split
train_df = df.sample(frac=0.9, random_state=42)
val_df = df.drop(train_df.index)
train_ds = AWGNDataset(train_df, cor_cols, orig_cols)
val_ds = AWGNDataset(val_df, cor_cols, orig_cols)

print('Train size:', len(train_ds), 'Val size:', len(val_ds))

# small dataloaders for testing
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# Print a sample shape
x0,y0 = train_ds[0]
print('Sample shapes x,y =', x0.shape, y0.shape)




# Model: NanoGPT-like transformer blocks (simplified) and a head

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout=0.0, resid_dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, n_heads, T, head_dim)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        out = att @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1,2).contiguous().reshape(B, T, C)
        out = self.out(out)
        out = self.resid_dropout(out)
        return out

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_hidden_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, attn_dropout=0.0, resid_dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * mlp_hidden_mult, dropout=dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPTForAWGN(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=100, mlp_mult=4, dropout=0.1):
        super().__init__()
        # We'll treat each input vector as a single "token" with embedding dimension d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, d_model))  # single position
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mlp_hidden_mult=mlp_mult, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, input_dim),
            nn.Tanh()  # map back to [-1,1]
        )
    def forward(self, x):
        # x: (B, seq_len, input_dim) but seq_len is 1 in our usage; we keep generic
        x = self.input_proj(x)
        x = x + self.pos_emb[:, :x.size(1), :]
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        x = self.head(x)
        return x

print('Model classes defined')








# Full GPU training pipeline across SNR stages
# Uses ErrorCorrectionDataset pattern from your example, prepares datasets per SNR and trains sequentially:

import os
from torch.utils.data import TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device', device)

# Configurable model/training params
SEQ_LEN = len(cor_cols)  # should be 4608
TRAIN_D_MODEL = 256
TRAIN_N_HEADS = 8
# Default number of layers for full model (kept configurable). Training below uses `train_layers` for speed.
DEFAULT_N_LAYERS = 100
train_layers = 100  # <--- small for quick training; change to 100 to match final architecture (requires large GPU/RAM)
mlp_mult = 4
dropout = 0.1

# Epoch schedule per SNR stage (one value per SNR in snr_db_pools order)
# You can tweak these. Using small defaults so the notebook run finishes in reasonable time.
snr_order = [1000, 500, 100, 50, 20, 10, 5]
epochs_per_stage = [100, 50, 20, 5, 5, 5, 5]  # editable
batch_size = 16
learning_rate = 2e-4

# Implement ErrorCorrectionDataset as in your example (returns tensors already on device)
class ErrorCorrectionDataset:
    def __init__(self, dataframe, snr_db_pools, train_size=0.8, seq_len=4608):
        self.df = dataframe
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.snr_db_pools = snr_db_pools
        self.train_size = train_size
        self.seq_len = seq_len
        self.corrupted_cols = [f"Corrupted_{i}" for i in range(seq_len)]
        self.original_cols = [f"Original_{i}" for i in range(seq_len)]
        self.X = dataframe[self.corrupted_cols].values.astype(np.float32)
        self.Y = dataframe[self.original_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        corrupted = self.X[idx]
        original = self.Y[idx]
        X = torch.tensor(corrupted, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(original, dtype=torch.float32)
        return X, y

    def _split_tensor(self, sub_df):
        n_train = int(self.train_size * len(sub_df))
        sub_df = sub_df.sample(frac=1, random_state=None).reset_index(drop=True)
        X = sub_df[self.corrupted_cols].values.astype(np.float32)
        y = sub_df[self.original_cols].values.astype(np.float32)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        return (
            torch.tensor(X_train, dtype=torch.float32).to(self.device),
            torch.tensor(y_train, dtype=torch.float32).to(self.device),
            torch.tensor(X_test, dtype=torch.float32).to(self.device),
            torch.tensor(y_test, dtype=torch.float32).to(self.device),
        )

    def prepare_datasets(self):
        datasets = []
        for snr_db in self.snr_db_pools:
            sub_df = self.df[self.df["SNR_DB"] == snr_db]
            datasets.append(self._split_tensor(sub_df))
        return datasets

    def generalized_dataset(self):
        return self._split_tensor(self.df)

# Prepare datasets using the order the user requested (start from high SNR)
snr_db_pools = snr_order
error_dataset_helper = ErrorCorrectionDataset(df, snr_db_pools, train_size=0.8, seq_len=SEQ_LEN)
error_datasets = error_dataset_helper.prepare_datasets()  # list of (X_train, y_train, X_test, y_test)
print('Prepared', len(error_datasets), 'datasets for SNRs:', snr_db_pools)

# Instantiate the model to train (on device). Use train_layers for actual training run below.
model = GPTForAWGN(input_dim=SEQ_LEN, d_model=TRAIN_D_MODEL, n_heads=TRAIN_N_HEADS, n_layers=train_layers, mlp_mult=mlp_mult, dropout=dropout).to(device)
print('Model created with', train_layers, 'layers and d_model=', TRAIN_D_MODEL)

# Optimizer, criterion, scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# criterion = nn.BCELoss()
criterion = nn.MSELoss()

# Mixed precision scaler if using CUDA
use_amp = (device.type == 'cuda')
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Utility: train for one stage
from tqdm import trange

def train_stage(model, X_train, y_train, X_val, y_val, epochs, batch_size, snr_label):
    model.train()
    ds = TensorDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    val_ds = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        n = 0
        for bx, by in loader:
            # bx: (B, seq_len), by: (B, seq_len)
            bx = bx.unsqueeze(1).to(device)  # (B,1,SEQ_LEN)
            by = by.unsqueeze(1).to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(bx)
                loss = criterion(out, by)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * bx.size(0)
            n += bx.size(0)
        avg_loss = total_loss / n if n else 0.0
        # validation
        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for vbx, vby in val_loader:
                vbx = vbx.unsqueeze(1).to(device)
                vby = vby.unsqueeze(1).to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    vout = model(vbx)
                    l = criterion(vout, vby)
                val_loss += l.item() * vbx.size(0)
                vn += vbx.size(0)
        val_loss = val_loss / vn if vn else 0.0
        model.train()
        if epoch % max(1, epochs//5) == 0 or epoch==1 or epoch==epochs:
            print(f"SNR={snr_label} | Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

# Create checkpoints dir
ckpt_dir = './checkpoints_gpt_awgn'
os.makedirs(ckpt_dir, exist_ok=True)

# Sequential training across SNR stages
for i, snr in enumerate(snr_db_pools):
    X_tr, y_tr, X_val, y_val = error_datasets[i]
    ep = epochs_per_stage[i] if i < len(epochs_per_stage) else epochs_per_stage[-1]
    print('\n' + '='*60)
    print(f'Starting training on SNR={snr} for {ep} epochs — dataset sizes: train={X_tr.size(0)}, val={X_val.size(0)}')
    train_stage(model, X_tr, y_tr, X_val, y_val, epochs=ep, batch_size=batch_size, snr_label=snr)
    # save checkpoint after stage
    ckpt_path = os.path.join(ckpt_dir, f'model_snr_{snr}_layers_{train_layers}.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f'Checkpoint saved to {ckpt_path}')

print('\nAll SNR stages complete. Final model saved.')
final_path = os.path.join(ckpt_dir, f'model_final_layers_{train_layers}.pth')
torch.save(model.state_dict(), final_path)
print('Final checkpoint:', final_path)






# quick inference check
model.eval()
with torch.no_grad():
    x_noisy, x_clean = val_ds[0]
    x_noisy = x_noisy.unsqueeze(0).to(device)
    out = model(x_noisy)
    print('out.shape', out.shape)
    print('sample clean vs pred (first token):')
    print(x_clean[0,:8])
    print(out[0,0,:8].cpu())