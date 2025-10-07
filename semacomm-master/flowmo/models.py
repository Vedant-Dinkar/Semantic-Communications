"""Model code for FlowMo.

Sources: https://github.com/feizc/FluxMusic/blob/main/train.py
https://github.com/black-forest-labs/flux/tree/main/src/flux
"""

import ast
import itertools
import math
from dataclasses import dataclass
from typing import List, Tuple

import einops
import torch
from einops import rearrange, repeat
from mup import MuReadout
from torch import Tensor, nn
import numpy as np
import lookup_free_quantize
from pyldpc import make_ldpc, encode, decode, get_message
import random
import os
import csv
import logging

MUP_ENABLED = True

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._guards").setLevel(logging.ERROR)

import torch.nn as nn


# New transformer-based error corrector matching the training code.
class AWGNErrorCorrector(nn.Module):
    def __init__(self, input_dim=288, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # Project input packed-code vector to model dimension
        self.input_fc = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: predict input_dim numbers per position (reconstruct packed codes)
        self.output_fc = nn.Linear(d_model, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x expected shape: [B, seq_len, input_dim] where input_dim==288
        x = self.input_fc(x)
        x = self.transformer_encoder(x)
        x = self.output_fc(x)
        return x

# # Path to the saved transformer-based error corrector.
# # Keep existing absolute path but allow an environment override.
# ERROR_CORRECTOR_PATH = os.environ.get(
#     "ERROR_CORRECTOR_PATH",
#     "/home/network/Documents/Semantic Communications/error_correction/error_corrector_updated_dataset.pth",
# )
ERROR_CORRECTOR_PATH = "/home/network/Documents/Semantic Communications/error_correction/code_awgn_error_corrector.pth"

def load_error_corrector(device: torch.device | None = None) -> nn.Module:
    """Load the transformer-based error corrector.

    The checkpoint is expected to contain the state_dict for an
    AWGNErrorCorrector trained to reconstruct packed-code vectors of
    dimension 288 per position. The returned model is moved to `device`
    and set to eval() mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading Error Corrector model from {ERROR_CORRECTOR_PATH}")

    # create model instance matching training architecture
    model = AWGNErrorCorrector(input_dim=288, d_model=256, nhead=8, num_layers=6)

    if not os.path.isfile(ERROR_CORRECTOR_PATH):
        raise FileNotFoundError(f"Error corrector checkpoint not found: {ERROR_CORRECTOR_PATH}")

    state = torch.load(ERROR_CORRECTOR_PATH, map_location=device)

    # handle cases where a dict contains other keys like {'model_state_dict': ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and next(iter(state)).startswith("module"):
        # try to strip a leading 'module.' if present
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        state = new_state

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
 
def predict_bytestream(bytestream: bytes, model: nn.Module | None = None, device: torch.device | None = None) -> bytes:
    """
    Corrects a corrupted bytestream using the trained error_corrector model.

    Args:
        bytestream (bytes): Corrupted bytestream (length 576 expected).
        model (torch.nn.Module, optional): Loaded model. If None, loads from disk.
        device (torch.device, optional): torch device. Auto-selects if None.

    Returns:
        bytes: Corrected bytestream.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = load_error_corrector(device)


    # Convert input bytes -> numpy -> tensor

    # interpret incoming bytestream as packed-code uint8s per position
    # Expected shapes:
    #  - single vector: length == 288 -> returns reconstructed 288 bytes
    #  - sequence: length % 288 == 0 -> reshaped to (seq_len, 288) and returned
    buf = np.frombuffer(bytestream, dtype=np.uint8)
    # Expect packed representation of shape (seq_len * code_dim) or (seq_len, input_dim)
    # Here dataset uses input_dim=288 per position. We'll try to reshape accordingly.
    if buf.size == 288:
        # single position vector
        arr = buf.astype(np.float32) / 255.0
        arr = arr.reshape(1, 1, 288)  # [B=1, seq_len=1, input_dim=288]
    elif buf.size == 288 * 1:
        arr = buf.astype(np.float32) / 255.0
        arr = arr.reshape(1, 1, 288)
    elif buf.size % 288 == 0:
        seq_len = buf.size // 288
        arr = buf.astype(np.float32).reshape(seq_len, 288) / 255.0
        arr = arr.reshape(1, seq_len, 288)
    else:
        raise ValueError(f"Input bytestream length {buf.size} is not compatible with input_dim=288")

    tensor_in = torch.tensor(arr, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(tensor_in)

    out_np = out.squeeze(0).cpu().numpy()  # [seq_len, 288]
    # convert back to uint8 packed representation
    out_flat = (out_np.reshape(-1) * 255.0).round().clip(0, 255).astype(np.uint8)
    return out_flat.tobytes()

error_corrector = load_error_corrector()

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    b, h, l, d = q.shape
    q, k = apply_rope(q, k, pe)

    if torch.__version__ == "2.0.1+cu117":  # tmp workaround
        if d != 64:
            print("MUP is broken in this setting! Be careful!")
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
            )
    else:
        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=8.0 / d if MUP_ENABLED else None,
        )
    assert x.shape == q.shape
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)],
        dim=-1,
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    # Safety check for empty tensors
    if xq.numel() == 0 or xk.numel() == 0:
        return xq, xk

    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def _get_diagonal_gaussian(parameters):
    mean, logvar = torch.chunk(parameters, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    return mean, logvar


def _sample_diagonal_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    x = mean + std * torch.randn(mean.shape, device=mean.device)
    return x


def _kl_diagonal_gaussian(mean, logvar):
    var = torch.exp(logvar)
    return 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar, dim=1).mean()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

        self.lin.weight[dim * 2 : dim * 3].data[:] = 0.0
        self.lin.bias[dim * 2 : dim * 3].data[:] = 0.0
        self.lin.weight[dim * 5 : dim * 6].data[:] = 0.0
        self.lin.bias[dim * 5 : dim * 6].data[:] = 0.0

    def forward(self, vec: Tensor) -> Tuple[ModulationOut, ModulationOut]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor):
        pe_single, pe_double = pe
        p = 1
        if vec is None:
            img_mod1, img_mod2 = ModulationOut(0, 1 - p, 1), ModulationOut(0, 1 - p, 1)
            txt_mod1, txt_mod2 = ModulationOut(0, 1 - p, 1), ModulationOut(0, 1 - p, 1)
        else:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (p + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (p + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe_double)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (p + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (p + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, txt


class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        readout_zero_init=False,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if MUP_ENABLED:
            self.linear = MuReadout(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                readout_zero_init=readout_zero_init,
            )
        else:
            self.linear = nn.Linear(
                hidden_size, patch_size * patch_size * out_channels, bias=True
            )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, vec) -> Tensor:
        if vec is None:
            pass
        else:
            shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
            x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.norm_final(x)
        x = self.linear(x)
        return x


@dataclass
class FluxParams:
    in_channels: int
    patch_size: int
    context_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    axes_dim: List[int]
    theta: int
    qkv_bias: bool


DIT_ZOO = dict(
    dit_xl_4=dict(
        hidden_size=1152,
        mlp_ratio=4.0,
        num_heads=16,
        axes_dim=[8, 28, 28],
        theta=10_000,
        qkv_bias=True,
    ),
    dit_l_4=dict(
        hidden_size=1024,
        mlp_ratio=4.0,
        num_heads=16,
        axes_dim=[8, 28, 28],
        theta=10_000,
        qkv_bias=True,
    ),
    dit_b_4=dict(
        hidden_size=768,
        mlp_ratio=4.0,
        num_heads=12,
        axes_dim=[8, 28, 28],
        theta=10_000,
        qkv_bias=True,
    ),
    dit_s_4=dict(
        hidden_size=384,
        mlp_ratio=4.0,
        num_heads=6,
        axes_dim=[8, 28, 28],
        theta=10_000,
        qkv_bias=True,
    ),
    dit_mup_test=dict(
        hidden_size=768,
        mlp_ratio=4.0,
        num_heads=12,
        axes_dim=[8, 28, 28],
        theta=10_000,
        qkv_bias=True,
    ),
)


def prepare_idxs(img, code_length, patch_size):
    bs, c, h, w = img.shape

    img_ids = torch.zeros(h // patch_size, w // patch_size, 3, device=img.device)
    img_ids[..., 1] = (
        img_ids[..., 1] + torch.arange(h // patch_size, device=img.device)[:, None]
    )
    img_ids[..., 2] = (
        img_ids[..., 2] + torch.arange(w // patch_size, device=img.device)[None, :]
    )
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt_ids = (
        torch.zeros((bs, code_length, 3), device=img.device)
        + torch.arange(code_length, device=img.device)[None, :, None]
    )
    return img_ids, txt_ids

def apply_latent_awgn_noise(code: torch.Tensor):
        """
        Apply Additive White Gaussian Noise (AWGN) to the latent code based on PSNR.

        Args:
            code (torch.Tensor): Input latent code tensor.
            noise_level (float): PSNR value (1-100 scale; higher = less noise).
            
        Returns:
            torch.Tensor: Noisy code tensor on the same device as input.
        """
        if not isinstance(code, torch.Tensor):
            raise TypeError("Code must be a torch.Tensor")
        
        print(torch.mean(code).item())
        noise_level = abs(random.random() * torch.mean(code).item())
        print(noise_level)

        # Automatically use CUDA if available and code is not already on CUDA
        device = code.device
        if torch.cuda.is_available() and not code.is_cuda:
            code = code.to("cuda")
            device = code.device

        # Handle edge cases
        if noise_level >= 100:
            return code.clone()  # Perfect channel: return unchanged

        if noise_level <= 0:
            raise ValueError("Noise level must be greater than 0")

        # Calculate signal power
        signal_power = torch.mean(code ** 2).item()

        # Convert PSNR to noise variance
        psnr_linear = 10 ** (noise_level / 10)
        noise_variance = signal_power / psnr_linear
        noise_std = noise_variance ** 0.5

        # Generate Gaussian noise on the same device
        noise = torch.randn_like(code, device=device) * noise_std

        # Add noise
        noisy_code = code + noise

        return noisy_code

def apply_bit_errors(code: torch.Tensor, ber: float = 1e-3) -> torch.Tensor:
    """
    Simulate random bit errors in the quantized code (bytestream).
    
    Args:
        code (torch.Tensor): Tensor of ints (quantized latent codes), shape [B, T, F]
        ber (float): Bit error rate (probability of flipping each bit).
    
    Returns:
        torch.Tensor: Noisy code tensor with bit flips applied.
    """
    if code.dtype == torch.bfloat16 or code.dtype == torch.float16:
        code = code.to(torch.float32)
    code_np = code.cuda().numpy()

    # Flatten to 1D byte array
    byte_arr = code_np.astype(np.int32).flatten()

    # Represent as uint32 bit patterns
    noisy_arr = []
    for val in byte_arr:
        noisy_val = val
        for bit in range(32):  # flip within 32 bits
            if random.random() < ber:
                noisy_val ^= (1 << bit)  # flip this bit
        noisy_arr.append(noisy_val)

    noisy_np = np.array(noisy_arr, dtype=np.int32).reshape(code_np.shape)
    noisy_code = torch.from_numpy(noisy_np).to(code.device)
    noisy_code = noisy_code.to(torch.bfloat16)
    return noisy_code

def pack_bits(x: torch.Tensor) -> torch.Tensor:
    """
    Pack a tensor of -1/+1 values into int16.
    Groups of 16 values become one int16.
    
    Args:
        x (torch.Tensor): Input tensor with values in {-1, +1}.
    
    Returns:
        torch.Tensor: Packed tensor of dtype int16.
    """
    # Flatten
    flat = x.flatten()
    assert flat.numel() % 16 == 0, "Number of elements must be divisible by 16."
    
    # Map -1 -> 0, +1 -> 1
    flat = (flat > 0).to(torch.int16)
    
    # Reshape into groups of 16
    flat = flat.view(-1, 16)
    
    # Precompute bit powers
    powers = (2 ** torch.arange(16, dtype=torch.int16, device=flat.device))
    
    # Pack bits into int16
    packed = (flat * powers).sum(dim=1).to(torch.int16)
    
    return packed


def unpack_bits(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    Unpack int16 tensor back into -1/+1 values.
    
    Args:
        packed (torch.Tensor): Packed int16 tensor.
        original_shape (torch.Size): Original shape of the data before packing.
    
    Returns:
        torch.Tensor: Unpacked tensor with values in {-1, +1}.
    """
    # Expand each int16 into 16 bits
    bits = ((packed.unsqueeze(1) >> torch.arange(16, device=packed.device)) & 1).to(torch.bfloat16)
    
    # Flatten
    flat = bits.view(-1)[:torch.prod(torch.tensor(original_shape))]
    
    # Map 0 -> -1, 1 -> +1
    flat = flat * 2 - 1
    
    return flat.view(original_shape)

def pack_bytestream(packed: torch.Tensor) -> bytes:
    """
    Convert packed int16 tensor -> raw bytes.
    """
    return packed.cpu().numpy().tobytes()


def unpack_bytestream(bytestream: bytes, device="cuda") -> torch.Tensor:
    """
    Convert raw bytes -> packed int16 tensor.
    """
    return torch.frombuffer(bytestream, dtype=torch.int16).to(device)

def corrupt_bytestream(bytestream: bytes, bit_error_rate: float = random.random() * 0.5) -> bytes:
    """
    Corrupt a bytestream by randomly flipping bits.

    Args:
        bytestream (bytes): Input bytestream.
        bit_error_rate (float): Probability of flipping each bit.

    Returns:
        bytes: Corrupted bytestream.
    """
    byte_array = bytearray(bytestream)
    n_bits = len(byte_array) * 8
    n_flips = int(n_bits * bit_error_rate)

    for _ in range(n_flips):
        # Pick random byte and bit
        byte_idx = random.randint(0, len(byte_array) - 1)
        bit_idx = random.randint(0, 7)
        # Flip the bit
        byte_array[byte_idx] ^= (1 << bit_idx)

    return bytes(byte_array)

def corrupt_bytestream_awgn(bytestream: bytes, psnr: float) -> bytes:
    """
    Corrupt a bytestream using AWGN, with proper BPSK modulation and PSNR control.

    Args:
        bytestream (bytes): Input bytestream.
        psnr (float): Target PSNR in dB (higher = less noise).

    Returns:
        bytes: Corrupted bytestream.
    """
    if psnr <= 0:
        raise ValueError("PSNR must be > 0")

    # Convert to bit array
    arr = np.frombuffer(bytestream, dtype=np.uint8)
    bits = np.unpackbits(arr)

    # Map bits {0,1} -> BPSK symbols {-1,+1}
    symbols = 2 * bits.astype(np.float32) - 1.0

    # Signal power
    signal_power = np.mean(symbols ** 2)  # should be ~1.0 for BPSK

    # Noise variance from PSNR
    noise_variance = signal_power / (10 ** (psnr / 10))
    noise_std = np.sqrt(noise_variance)

    # Add Gaussian noise
    noisy_symbols = symbols + np.random.normal(0, noise_std, size=symbols.shape)

    # Hard decision back to bits
    noisy_bits = (noisy_symbols > 0).astype(np.uint8)

    # Pack bits back into bytes
    noisy_arr = np.packbits(noisy_bits)

    return noisy_arr.tobytes()


def add_awgn_to_bytestream(bytestream, snr_db):
    """
    Adds Additive White Gaussian Noise (AWGN) to a bytestream.
    
    Args:
        bytestream (bytes): The input bytestream.
        snr_db (float): The desired Signal-to-Noise Ratio in decibels (dB).
        
    Returns:
        bytes: The output bytestream with added AWGN.
    """
    # 1. Convert bytestream to a numerical NumPy array
    # We use uint8 and normalize to a float range, e.g., [-1, 1] or [0, 1]
    signal_array = np.frombuffer(bytestream, dtype=np.uint8).astype(np.float64)
    
    # Normalize the signal to the range [-1, 1] for a cleaner power calculation
    signal_normalized = (signal_array / 127.5) - 1.0
    
    # 2. Calculate the signal power
    # Signal power is the mean of the squared signal values
    signal_power = np.mean(signal_normalized**2)
    
    # 3. Convert SNR from dB to a linear scale
    snr_linear = 10**(snr_db / 10.0)
    
    # Calculate the noise power
    # SNR_linear = Signal_Power / Noise_Power -> Noise_Power = Signal_Power / SNR_linear
    if snr_linear <= 0:
        raise ValueError("SNR must be greater than 0")
    noise_power = signal_power / snr_linear
    
    # Calculate the noise standard deviation
    noise_std = np.sqrt(noise_power)
    
    # 4. Generate AWGN with the calculated standard deviation
    noise = noise_std * np.random.randn(len(signal_normalized))
    
    # 5. Add the noise to the normalized signal
    noisy_signal_normalized = signal_normalized + noise
    
    # 6. Convert the noisy signal back to bytes
    # Scale and clamp the values to the original uint8 range [0, 255]
    noisy_signal_array = ((noisy_signal_normalized + 1.0) * 127.5)
    noisy_signal_array = np.clip(noisy_signal_array, 0, 255).astype(np.uint8)
    
    return noisy_signal_array.tobytes()


def build_and_save_dataset(bytestream, n_samples=10, 
                           ber_pools=[0.0, 0.2, 0.5, 1.0], 
                           save_path="./error_correction/dataset.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    byte_len = len(bytestream)

    with open(save_path, "a+", newline="") as f:
        writer = csv.writer(f)

        # # header: BER + Original bytes + Corrupted bytes
        # header = ["BER"] + [f"Original_{i}" for i in range(byte_len)] + [f"Corrupted_{i}" for i in range(byte_len)]
        # writer.writerow(header)

        for _ in range(n_samples):
            original = list(bytestream)
            for ber in ber_pools:
                corrupted = list(corrupt_bytestream(bytestream, ber))
                row = [ber] + original + corrupted
                writer.writerow(row)

    print(f"Dataset saved to {save_path}, total rows={n_samples * len(ber_pools)}")

def build_and_save_dataset_awgn(bytestream, n_samples=10, 
                           psnr_pools=[1, 5, 10, 20, 50], 
                           save_path="./error_correction/awgn_dataset.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    byte_len = len(bytestream)

    with open(save_path, "a+", newline="") as f:
        writer = csv.writer(f)

        # header: Noise + Original bytes + Corrupted bytes
        # header = ["PSNR"] + [f"Original_{i}" for i in range(byte_len)] + [f"Corrupted_{i}" for i in range(byte_len)]
        # writer.writerow(header)

        for _ in range(n_samples):
            original = list(bytestream)
            for psnr in psnr_pools:
                corrupted = list(corrupt_bytestream_awgn(bytestream, psnr))
                row = [psnr] + original + corrupted
                writer.writerow(row)

    print(f"Dataset saved to {save_path}, total rows={n_samples * len(psnr_pools)}")

def create_dataset_awgn(bytestream, n_samples=10, 
                           snr_db_pools=[5, 10, 20, 50, 100, 500, 1000], 
                           save_path="./error_correction/code_awgn_dataset.csv", shape = [-1, -1, -1]):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    original_code = unpack_bytestream(bytestream)
    # original_code = unpack_bits(original_code, original_shape=shape)
    # final_code = unpack_bits(original_code, original_shape=shape)   
    # print(final_code)
    if isinstance(original_code, torch.Tensor):
        original_code = original_code.detach().cpu().numpy().flatten()
    else:
        original_code = list(original_code)
    code_len = len(original_code)

    with open(save_path, "a+", newline="") as f:
        writer = csv.writer(f)

        # header: Noise + Original bytes + Corrupted bytes
        # header = ["SNR_DB"] + [f"Original_{i}" for i in range(code_len)] + [f"Corrupted_{i}" for i in range(code_len)]
        # writer.writerow(header)

        for _ in range(n_samples):
            original = list(bytestream)
            for snr_db in snr_db_pools:
                corrupted = add_awgn_to_bytestream(bytestream, snr_db)
                corrupted_code = unpack_bytestream(corrupted)

                if isinstance(corrupted_code, torch.Tensor):
                    corrupted_code = corrupted_code.detach().cpu().numpy().flatten()
                else:
                    corrupted_code = list(corrupted_code)

                row = [snr_db] + list(original_code) + list(corrupted_code)
                writer.writerow(row)

class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams, name="", lsg=False):
        super().__init__()

        self.name = name
        self.lsg = lsg
        self.params = params
        self.in_channels = params.in_channels
        self.patch_size = params.patch_size
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.txt_in = nn.Linear(params.context_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for idx in range(params.depth)
            ]
        )

        self.final_layer_img = LastLayer(
            self.hidden_size, 1, self.out_channels, readout_zero_init=False
        )
        self.final_layer_txt = LastLayer(
            self.hidden_size, 1, params.context_dim, readout_zero_init=False
        )

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        b, c, h, w = img.shape

        img = rearrange(
            img,
            "b c (gh ph) (gw pw) -> b (gh gw) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        img = self.img_in(img)

        if timesteps is None:
            vec = None
        else:
            vec = self.time_in(timestep_embedding(timesteps, 256))

        txt = self.txt_in(txt)
        pe_single = self.pe_embedder(torch.cat((txt_ids,), dim=1))
        pe_double = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, pe=(pe_single, pe_double), vec=vec)

        img = self.final_layer_img(img, vec=vec)
        img = rearrange(
            img,
            "b (gh gw) (ph pw c) -> b c (gh ph) (gw pw)",
            ph=self.patch_size,
            pw=self.patch_size,
            gh=h // self.patch_size,
            gw=w // self.patch_size,
        )

        txt = self.final_layer_txt(txt, vec=vec)
        return img, txt, {"final_txt": txt}


def get_weights_to_fix(model):
    with torch.no_grad():
        for name, module in itertools.chain(model.named_modules()):
            if "double_blocks" in name and isinstance(module, torch.nn.Linear):
                yield name, module.weight


class FlowMo(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        code_length = config.model.code_length
        context_dim = config.model.context_dim
        enc_depth = config.model.enc_depth
        dec_depth = config.model.dec_depth

        patch_size = config.model.patch_size
        self.config = config

        self.image_size = config.data.image_size
        self.patch_size = config.model.patch_size
        self.code_length = code_length
        self.dit_mode = "dit_b_4"
        self.context_dim = context_dim
        self.encoder_context_dim = context_dim * (
            1 + (self.config.model.quantization_type == "kl")
        )

        if config.model.quantization_type == "lfq":
            self.quantizer = lookup_free_quantize.LFQ(
                codebook_size=2**self.config.model.codebook_size_for_entropy,
                dim=self.config.model.codebook_size_for_entropy,
                num_codebooks=1,
                token_factorization=False,
            )

        if self.config.model.enc_mup_width is not None:
            enc_width = self.config.model.enc_mup_width
        else:
            enc_width = width

        encoder_params = FluxParams(
            in_channels=3 * patch_size**2,
            context_dim=self.encoder_context_dim,
            patch_size=patch_size,
            depth=enc_depth,
            **DIT_ZOO[self.dit_mode],
        )
        decoder_params = FluxParams(
            in_channels=3 * patch_size**2,
            context_dim=context_dim + 1,
            patch_size=patch_size,
            depth=dec_depth,
            **DIT_ZOO[self.dit_mode],
        )

        # width=4, dit_b_4 is the usual model
        encoder_params.hidden_size = enc_width * (encoder_params.hidden_size // 4)
        decoder_params.hidden_size = width * (decoder_params.hidden_size // 4)
        encoder_params.axes_dim = [
            (d // 4) * enc_width for d in encoder_params.axes_dim
        ]
        decoder_params.axes_dim = [(d // 4) * width for d in decoder_params.axes_dim]

        self.encoder = Flux(encoder_params, name="encoder")
        self.decoder = Flux(decoder_params, name="decoder")

    # @torch.compile
    def encode(self, img):
        b, c, h, w = img.shape

        img_idxs, txt_idxs = prepare_idxs(img, self.code_length, self.patch_size)
        txt = torch.zeros(
            (b, self.code_length, self.encoder_context_dim), device=img.device
        )

        _, code, aux = self.encoder(img, img_idxs, txt, txt_idxs, timesteps=None)

        return code, aux

    def _decode(self, img, code, timesteps):
        b, c, h, w = img.shape

        img_idxs, txt_idxs = prepare_idxs(
            img,
            self.code_length,
            self.patch_size,
        )
        pred, _, decode_aux = self.decoder(
            img, img_idxs, code, txt_idxs, timesteps=timesteps
        )
        return pred, decode_aux

    # @torch.compile
    def decode(self, *args, **kwargs):
        return self._decode(*args, **kwargs)

    # @torch.compile
    def decode_checkpointed(self, *args, **kwargs):
        # Need to compile(checkpoint), not checkpoint(compile)
        assert not kwargs, kwargs
        return torch.utils.checkpoint.checkpoint(
            self._decode,
            *args,
            # WARNING: Do not use_reentrant=True with compile, it will silently
            # produce incorrect gradients!
            use_reentrant=False,
        )

    @torch.compile
    def _quantize(self, code):
        """
        Args:
            code: [b codelength context dim]

        Returns:
            quantized code of the same shape
        """
        # noisy_code = apply_latent_awgn_noise(code)
        # code = noisy_code
        print("Quantization Type: ", self.config.model.quantization_type)
        print("Before quantization for shape: ", code.shape)
        print(code)
        org_code = code
        quantized_code = code
        print("-" * 30)
        b, t, f = code.shape
        indices = None
        if self.config.model.quantization_type == "noop":
            quantized = code
            quantized_code = quantized
            quantizer_loss = torch.tensor(0.0).to(code.device)
        elif self.config.model.quantization_type == "kl":
            # colocating features of same token before split is maybe slightly
            # better?
            mean, logvar = _get_diagonal_gaussian(
                einops.rearrange(code, "b t f -> b (f t)")
            )
            code = einops.rearrange(
                _sample_diagonal_gaussian(mean, logvar),
                "b (f t) -> b t f",
                f=f // 2,
                t=t,
            )
            quantizer_loss = _kl_diagonal_gaussian(mean, logvar)
        elif self.config.model.quantization_type == "lfq":
            assert f % self.config.model.codebook_size_for_entropy == 0, f
            code = einops.rearrange(
                code,
                "b t (fg fh) -> b fg (t fh)",
                fg=self.config.model.codebook_size_for_entropy,
            )

            (quantized, entropy_aux_loss, indices), breakdown = self.quantizer(
                code, return_loss_breakdown=True
            )
            assert quantized.shape == code.shape
            quantized = einops.rearrange(quantized, "b fg (t fh) -> b t (fg fh)", t=t)

            quantizer_loss = (
                entropy_aux_loss * self.config.model.entropy_loss_weight
                + breakdown.commitment * self.config.model.commit_loss_weight
            )
            code = quantized
            quantized_code = quantized
        else:
            raise NotImplementedError

        print("After quantization for shape: ", quantized_code.shape)
        print(quantized_code)
        print("-" * 30)

        b, t, f = code.shape
        print("Before and after noise for shape: ", code.shape)
        print("Original:", code.shape)

        # Pack
        code = pack_bits(code)
        print("Packed:", code.shape, code.dtype)

        orginal_code = code

        # Convert to bytestream
        bytestream = pack_bytestream(code)
        print("Bytestream length:", len(bytestream))

        # Corrupt bytestream
        # corrupted_bytestream = corrupt_bytestream(bytestream, bit_error_rate=0.1)
        # corrupt_bytestream = bytestream
        # ratio = random.random()   # PSNR between 0 and 20 dB
        # corrupted_bytestream = corrupt_bytestream_awgn(bytestream, ratio)
        # print("Ratio:", ratio)

        # build_and_save_dataset_awgn(bytestream, n_samples=5, 
        #                            psnr_pools=[0.025, 0.1, 1, 2, 5, 50], 
        #                            save_path="./error_correction/awgn_dataset.csv")
        # print("Corrupted bytestream length:", len(corrupted_bytestream))

        # Recover bytestream
        # bytestream = predict_bytestream(corrupted_bytestream)
        # bytestream = corrupted_bytestream

        # build_and_save_dataset(bytestream, n_samples=5, 
        #                        ber_pools=[0.0, 0.2, 0.5, 0.8], 
        #                        save_path="./error_correction/dataset.csv")


        # raise RuntimeError("Stop here for dataset generation")
        # print("Recovered bytestream length:", len(bytestream))

        
        # create_dataset_awgn(bytestream, n_samples=5, 
        #                            snr_db_pools=[5, 10, 20, 50, 100, 500, 1000], 
        #                            save_path="./error_correction/code_awgn_dataset.csv", shape = [b, t, f])
    
        # raise RuntimeError("Stop here for dataset generation")

        corrupted_bytestream = add_awgn_to_bytestream(bytestream, snr_db=500)
        print("Bytestream length after AWGN:", len(bytestream))

        # Unpack bytestream
        corrupted_code = unpack_bytestream(corrupted_bytestream)
        print("Unpacked from bytes:", code.shape, code.dtype)

        try:
            # corrupted_code is a torch tensor of dtype int16 representing packed words
            device_pred = None
            if 'error_corrector' in globals() and error_corrector is not None:
                # model device
                try:
                    device_pred = next(error_corrector.parameters()).device
                except Exception:
                    device_pred = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device_pred = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Move corrupted_code to cpu numpy for normalization
            if isinstance(corrupted_code, torch.Tensor):
                # ensure conversion to a numpy-friendly dtype
                corrupted_np = corrupted_code.detach().cpu().to(torch.float32).numpy().astype(np.float32)
            else:
                corrupted_np = np.array(corrupted_code, dtype=np.float32)

            # Normalization parameters (matches notebook dataset preprocessing)
            min_val = -32768.0
            max_val = 32767.0

            # Normalize to [0,1]
            norm = (corrupted_np - min_val) / (max_val - min_val)

            # Try to reshape into a [B, seq_len, 288] input for the transformer
            total = norm.size
            reshaped = None
            if total == (b * t * 288):
                reshaped = norm.reshape(b, t, 288)
            elif total % 288 == 0:
                seq_len = total // 288
                reshaped = norm.reshape(1, seq_len, 288)
            else:
                # fallback: reshape to [b, t, f] if shapes match
                if total == (b * t * f):
                    reshaped = norm.reshape(b, t, f)
                else:
                    # As last resort, flatten to single sequence of length 288 chunks
                    seq_len = max(1, total // 288)
                    trimmed = norm[: seq_len * 288]
                    reshaped = trimmed.reshape(1, seq_len, 288)

            tensor_in = torch.tensor(reshaped, dtype=torch.float32, device=device_pred)

            with torch.no_grad():
                pred_out = error_corrector(tensor_in)

            # ensure predicted tensor is float32 on cpu before converting to numpy
            pred_np = pred_out.cpu().to(torch.float32).numpy().reshape(-1)[: total]

            # Denormalize back to int16 range
            denorm = np.round(pred_np * (max_val - min_val) + min_val).astype(np.int16)

            # Ensure same length as corrupted_np; if we trimmed earlier, pad/trim
            if denorm.size != corrupted_np.size:
                denorm = np.resize(denorm, corrupted_np.shape)

            # Write back into corrupted_code as torch tensor of dtype int16 on original device
            corrupted_code = torch.from_numpy(denorm).to(code.device)
        except Exception as e:
            # If prediction fails, fall back to using the noisy corrupted_code
            print("Error during error-corrector prediction:", e)
            # keep corrupted_code as-is

        # Unpack to restore shape [b, t, f]
        restored = unpack_bits(corrupted_code, [b, t, f])
        # ensure dtype/device match original quantized tensor
        try:
            restored = restored.to(quantized_code.dtype).to(quantized_code.device)
        except Exception:
            restored = restored.to(quantized_code.device)

        # assign back to `code` so callers receive the restored [b,t,f] tensor
        code = restored
        print("Restored equals original?", torch.equal(code, quantized_code))

        # code = code.to(torch.bfloat16)
        print("-" * 30)
        # noisy_code = apply_latent_awgn_noise(code)
        # save_tensor_to_csv(code, noisy_code)
        # Save code vector to CSVs
       
        # ======= Aadish changes ==========
        # code_np = code.detach().cpu().numpy()
        # csv_path = os.path.join(os.getcwd(), "quantized_codes.csv")
        # with open(csv_path, "a", newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for code_vec in code_np.reshape(-1, code_np.shape[-1]):
        #         writer.writerow(code_vec)
        # ======= Aadish changes ==========

        return code, indices, quantizer_loss

    def forward(
        self,
        img,
        noised_img,
        timesteps,
        enable_cfg=True,
    ):
        aux = {}

        code, encode_aux = self.encode(img)

        aux["original_code"] = code

        b, t, f = code.shape

        code, _, aux["quantizer_loss"] = self._quantize(code)
        noisy_code = apply_latent_awgn_noise(code)

        mask = torch.ones_like(noisy_code[..., :1])
        # code = torch.concatenate([code, mask], axis=-1)
        noisy_code = torch.cat([code, mask], dim=-1)
        code_pre_cfg = noisy_code

        print("Forward Pass...")

        if self.config.model.enable_cfg and enable_cfg:
            cfg_mask = (torch.rand((b,), device=code.device) > 0.1)[:, None, None]
            noisy_code = noisy_code * cfg_mask

        # v_est, decode_aux = self.decode(noised_img, code, timesteps)
        v_est, decode_aux = self.decode_checkpointed(noised_img, noisy_code, timesteps)

        aux.update(decode_aux)

        if self.config.model.posttrain_sample:
            aux["posttrain_sample"] = self.reconstruct_checkpoint(code_pre_cfg)

        return v_est, aux

    def reconstruct_checkpoint(self, code):
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            bs, *_ = code.shape

            z = torch.randn((bs, 3, self.image_size, self.image_size)).cuda()
            ts = torch.rand((bs, self.config.model.posttrain_sample_k + 1)).cumsum(dim=1).cuda()
            ts = ts - ts[:, :1]
            ts = (ts / ts[:, -1:]).flip(dims=(1,))
            dts = ts[:, :-1] - ts[:, 1:]

            for i, (t, dt) in enumerate((zip(ts.T, dts.T))):
                if self.config.model.posttrain_sample_enable_cfg:
                    mask = (torch.rand((bs,), device=code.device) > 0.1)[
                        :, None, None
                    ].to(code.dtype)
                    code_t = code * mask
                else:
                    code_t = code

                vc, _ = self.decode_checkpointed(z, code_t, t)

                z = z - dt[:, None, None, None] * vc
        return z

    @torch.no_grad()
    def reconstruct(self, images, dtype=torch.bfloat16, code=None):
        """
        Args:
            images in [bchw] [-1, 1]

        Returns:
            images in [bchw] [-1, 1]
        """
        model = self
        config = self.config.eval.sampling

        print("Reconstruct...")

        with torch.autocast(
            "cuda",
            dtype=dtype,
        ):
            bs, c, h, w = images.shape
            if code is None:
                x = images.cuda()
                prequantized_code = model.encode(x)[0].cuda()
                code, _, _ = model._quantize(prequantized_code)

            z = torch.randn((bs, 3, h, w)).cuda()

            mask = torch.ones_like(code[..., :1])
            code = torch.concatenate([code * mask, mask], axis=-1)

            cfg_mask = 0.0
            null_code = code * cfg_mask if config.cfg != 1.0 else None

            

            samples = rf_sample(
                model,
                z,
                code,
                null_code=null_code,
                sample_steps=config.sample_steps,
                cfg=config.cfg,
                schedule=config.schedule,
            )[-1].clip(-1, 1)
        return samples.to(torch.float32)
    

    @torch.no_grad()
    def reconstruct_noise(self, images, Noise_level=100, dtype=torch.bfloat16, code=None):
        """
        Args:
            images in [bchw] [-1, 1]

        Returns:
            images in [bchw] [-1, 1]
        """
        model = self
        config = self.config.eval.sampling

        print("Reconstruct Noise...")

        with torch.autocast(
            "cuda",
            dtype=dtype,
        ):
            bs, c, h, w = images.shape
            if code is None:
                x = images.cuda()
                print("Input Dimensions: ", x.shape)
                print(x)
                print("-" * 30)
                prequantized_code = model.encode(x)[0].cuda()
                print("Prequantized Dimensions: ", prequantized_code.shape)
                print(prequantized_code)
                print("-" * 30)

                code, _, _ = model._quantize(prequantized_code)

            z = torch.randn((bs, 3, h, w)).cuda()

            # print(code.shape)
            # print(code)
            # APPLY AWGN NOISE without encoding
            code = apply_awgn_noise(code, device=code.device)
    
            # instead of direct AWGN, we LDPC-protect first
            # code = ldpc_protect_and_send(code, snr_db=Noise_level)

            mask = torch.ones_like(code[..., :1])
            code = torch.concatenate([code * mask, mask], axis=-1)

            cfg_mask = 0.0
            null_code = code * cfg_mask if config.cfg != 1.0 else None

            print("Code...")
            print(code)
            print("-" * 30)
            samples = rf_sample(
                model,
                z,
                code,
                null_code=null_code,
                sample_steps=config.sample_steps,
                cfg=config.cfg,
                schedule=config.schedule,
            )[-1].clip(-1, 1)
        return samples.to(torch.float32)


def rf_loss(config, model, batch, aux_state):
    x = batch["image"]
    b = x.size(0)

    if config.opt.schedule == "lognormal":
        nt = torch.randn((b,)).to(x.device)
        t = torch.sigmoid(nt)
    elif config.opt.schedule == "fat_lognormal":
        nt = torch.randn((b,)).to(x.device)
        t = torch.sigmoid(nt)
        t = torch.where(torch.rand_like(t) <= 0.9, t, torch.rand_like(t))
    elif config.opt.schedule == "uniform":
        t = torch.rand((b,), device=x.device)
    elif config.opt.schedule.startswith("debug"):
        p = float(config.opt.schedule.split("_")[1])
        t = torch.ones((b,), device=x.device) * p
    else:
        raise NotImplementedError

    t = t.view([b, *([1] * len(x.shape[1:]))])
    z1 = torch.randn_like(x)
    zt = (1 - t) * x + t * z1

    zt, t = zt.to(x.dtype), t.to(x.dtype)

    vtheta, aux = model(
        img=x,
        noised_img=zt,
        timesteps=t.reshape((b,)),
    )

    diff = z1 - vtheta - x
    x_pred = zt - vtheta * t

    loss = ((diff) ** 2).mean(dim=list(range(1, len(x.shape))))
    loss = loss.mean()

    aux["loss_dict"] = {}
    aux["loss_dict"]["diffusion_loss"] = loss
    aux["loss_dict"]["quantizer_loss"] = aux["quantizer_loss"]

    if config.opt.lpips_weight != 0.0:
        aux_loss = 0.0
        if config.model.posttrain_sample:
            x_pred = aux["posttrain_sample"]

        lpips_dist = aux_state["lpips_model"](x, x_pred)
        lpips_dist = (config.opt.lpips_weight * lpips_dist).mean() + aux_loss
        aux["loss_dict"]["lpips_loss"] = lpips_dist
    else:
        lpips_dist = 0.0

    loss = loss + aux["quantizer_loss"] + lpips_dist
    aux["loss_dict"]["total_loss"] = loss
    return loss, aux


def _edm_to_flow_convention(noise_level):
    # z = x + \sigma z'
    return noise_level / (1 + noise_level)


def rf_sample(
    model,
    z,
    code,
    null_code=None,
    sample_steps=25,
    cfg=2.0,
    schedule="linear",
):
    b = z.size(0)
    if schedule == "linear":
        ts = torch.arange(1, sample_steps + 1).flip(0) / sample_steps
        dts = torch.ones_like(ts) * (1.0 / sample_steps)
    elif schedule.startswith("pow"):
        p = float(schedule.split("_")[1])
        ts = torch.arange(0, sample_steps + 1).flip(0) ** (1 / p) / sample_steps ** (
            1 / p
        )
        dts = ts[:-1] - ts[1:]
    else:
        raise NotImplementedError

    if model.config.eval.sampling.cfg_interval is None:
        interval = None
    else:
        cfg_lo, cfg_hi = ast.literal_eval(model.config.eval.sampling.cfg_interval)
        interval = _edm_to_flow_convention(cfg_lo), _edm_to_flow_convention(cfg_hi)

    images = []
    for i, (t, dt) in enumerate((zip(ts, dts))):
        timesteps = torch.tensor([t] * b).to(z.device)
        vc, decode_aux = model.decode(
            img=z, timesteps=timesteps, code=code
        )

        if null_code is not None and (
            interval is None
            or ((t.item() >= interval[0]) and (t.item() <= interval[1]))
        ):
            vu, _ = model.decode(img=z, timesteps=timesteps, code=null_code)
            vc = vu + cfg * (vc - vu)

        z = z - dt * vc
        images.append(z)
    return images


def apply_awgn_noise(code, device=None):
    """
    Apply Additive White Gaussian Noise (AWGN) to the latent code based on PSNR.
    
    Args:
        code: Input tensor (latent code)
        device: Optional device specification. If None, uses the device of the input tensor
    
    Returns:
        Noisy code tensor on the same device as input
    """
    return code
    # Ensure code is a tensor
    if not isinstance(code, torch.Tensor):
        raise TypeError("Code must be a torch.Tensor")
    
    # Use input tensor's device if not specified
    if device is None:
        device = code.device
    
    # Move to specified device if needed
    code = code.to(device)
    
    noise_level = random.random() * 70 + 5
    print(noise_level)

    # Handle edge cases
    if noise_level >= 100:
        # Perfect channel, no noise
        return code
    
    # if noise_level <= 0:
    #     # Invalid noise level
    #     raise ValueError("Noise level must be positive")
    
    # Calculate signal power (average power per element)
    # Using torch operations for efficiency
    signal_power = torch.mean(code ** 2).item()
    
    # Convert PSNR to noise variance
    # PSNR = 10 * log10(P / σ²) where P is signal power
    # σ² = P / 10^(PSNR/10)
    psnr_linear = 10 ** (noise_level / 10)
    noise_variance = signal_power / psnr_linear
    noise_std = np.sqrt(noise_variance)
    
    # Generate noise with same shape as code, directly on the target device
    noise = torch.randn_like(code, device=device) * noise_std
    
    # Add noise to the code
    noisy_code = code + noise
    
    return noisy_code



n = 512     # codeword length
dv, dc = 2, 4
H, G = make_ldpc(n, dv, dc, systematic=True, sparse=True)
k = G.shape[1]     # message length

def ldpc_protect_and_send(code: torch.Tensor,
                          snr_db: float) -> torch.Tensor:
    """
    1) Quantize float‐tensor -> uint8
    2) Unpack bits, pad/truncate to multiple of k
    3) LDPC‐encode each k‐bit message -> n‐bit codeword
    4) BPSK, AWGN(snr_db)
    5) LLR-> LDPC‐decode -> message bits -> reassemble bytes -> floats
    """
    device = code.device
    # --- 1) quantize to uint8 in [0,255] ---
    code_clamped = code.clamp(-1,1)
    code_u8 = (((code_clamped + 1) / 2) * 255).round().to(torch.uint8)
    # flatten batch
    bs = code_u8.shape[0]
    flat = code_u8.view(bs, -1).cpu().numpy()        # shape [B, Nbytes]
    # unpack to bits => shape [B, Nbits]
    bits = np.unpackbits(flat, axis=1)
    Nbits = bits.shape[1]
    # pad to multiple of k
    pad = (-Nbits) % k
    if pad:
        bits = np.pad(bits, ((0,0),(0,pad)), 'constant')
    # reshape into [B, #blocks, k]
    messages = bits.reshape(bs, -1, k)
    
    # --- 3) LDPC encode block-by-block ---
    codewords = []
    for b in range(bs):
        cw_b = []
        for msg in messages[b]:
            # Use proper pyldpc encode function: encode(G, v, snr)
            # For encoding, we use a high SNR (no noise during encoding)
            cw = encode(G, msg, snr=100)   # High SNR for clean encoding
            cw_b.append(cw)
        codewords.append(np.stack(cw_b))
    codewords = np.stack(codewords)       # [B, #blocks, n]

    # --- 4) BPSK and AWGN ---
    x = 1 - 2*codewords                  # 0->+1,1->-1
    snr_lin = 10**(snr_db/10)
    sigma = np.sqrt(1/(2*snr_lin))  # Corrected for BPSK
    noise = sigma * np.random.randn(*x.shape)
    y = x + noise

    # --- 5) decode LDPC and reassemble ---
    decoded_msgs = []
    for b in range(bs):
        dm_b = []
        for i in range(codewords.shape[1]):
            # Use proper pyldpc decode function: decode(H, y, snr)
            decoded_bits = decode(H, y[b,i], snr=snr_db, maxiter=50)
            # Extract message bits (first k bits for systematic code)
            msg_est = decoded_bits[:k]
            dm_b.append(msg_est)
        decoded_msgs.append(np.stack(dm_b))
    decoded_msgs = np.stack(decoded_msgs)  # [B, #blocks, k]
    decoded_bits = decoded_msgs.reshape(bs, -1)[:, :Nbits]

    # pack bits->bytes->uint8 tensor
    out_bytes = np.packbits(decoded_bits, axis=1)
    out_bytes = out_bytes.reshape(code_u8.shape)   # [B, ...]
    code_rec = torch.from_numpy(out_bytes).to(device).to(torch.float32)
    # dequantize back to [-1,1]
    code_rec = (code_rec / 255)*2 - 1
    return code_rec