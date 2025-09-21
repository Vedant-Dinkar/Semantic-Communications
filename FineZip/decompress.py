#!/usr/bin/env python3
"""
decompress.py
Reads .finezip file produced by compress.py, reconstructs tokens using the same base model + tokenizer,
and writes the decoded text (UTF-8) to the output file.

Usage:
    python decompress.py --in out.finezip --out out_decoded.txt --device cpu --model gpt2
    python3 decompress.py --in "mytext - {index}.finezip" --out "decoded - {index}.txt" --device cuda
"""
import argparse
import bz2
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import io

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True, help="Input .finezip file produced by compress.py")
    p.add_argument("--out", required=True, help="Output decoded text file")
    p.add_argument("--model", default="gpt2", help="HF model id used for decoding (must match compress run tokenizer/model)")
    p.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else args.device)

    print("Loading compressed file:", args.infile)
    with open(args.infile, "rb") as f:
        compressed = f.read()
    raw = bz2.decompress(compressed)

    # Load npz from bytes
    npzfile = io.BytesIO(raw)
    with np.load(npzfile, allow_pickle=True) as data:
        first_tokens = data["first_tokens"].astype(np.uint32)
        ranks = data["ranks"].astype(np.uint32)
        chunk_lengths = data["chunk_lengths"].astype(np.uint32)
        metadata_json = data["metadata_json"].tolist()
        try:
            metadata = json.loads(metadata_json)
        except:
            metadata = {}
    print("Metadata:", metadata.get("chunk_size"))

    # Load tokenizer & model (must be same type as used during compression)
    print("Loading tokenizer & model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    vocab_size = metadata.get("vocab_size", tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else tokenizer.vocab_size)
    print("Vocab size:", vocab_size)

    # Reconstruct tokens chunk by chunk
    reconstructed_token_ids = []
    rank_ptr = 0
    for chunk_idx, L in enumerate(tqdm(chunk_lengths, desc="Chunks")):
        if L == 0:
            continue
        first_tok = int(first_tokens[chunk_idx])
        chunk_ids = [first_tok]
        if L == 1:
            reconstructed_token_ids.extend(chunk_ids)
            continue

        # We'll reconstruct positions 1..L-1 sequentially
        for pos in range(1, L):
            # Prepare prefix (tokens up to current pos-1) as input to model
            prefix = torch.tensor([chunk_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                outs = model(prefix)
                # logits shape [1, prefix_len, V]; we need logits that predict next token -> logits[0, -1]
                logits = outs.logits[0, -1].cpu()

            rank = int(ranks[rank_ptr])
            rank_ptr += 1

            # pick token that is at position (rank-1) in descending order of logits
            # we compute argsort descending and pick index rank-1
            # Note: for large vocab this costs O(V log V) but vocab for gpt2 (~50k) is fine for small files
            sorted_idx = torch.argsort(logits, descending=True)
            selected_token = int(sorted_idx[rank-1].item())
            chunk_ids.append(selected_token)

        reconstructed_token_ids.extend(chunk_ids)

    # Detokenize
    decoded_text = tokenizer.decode(reconstructed_token_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(decoded_text)
    print("Wrote decoded text to:", args.out)

if __name__ == "__main__":
    main()
