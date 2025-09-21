#!/usr/bin/env python3
"""
compress.py
Simple FineZip-style rank + compressor pipeline using an open model (gpt2).
Produces a single output file <out_path>.finezip (bz2-compressed .npz)

Usage:
    python compress.py --input input.txt --out out.finezip --model gpt2 --chunk_size 512 --batch 8 --device cpu
    python3 compress.py --input "input - {index}.txt" --out "myfile - {index}" --device cuda
"""
import argparse
import json
import os
import bz2
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input UTF-8 text file to compress")
    p.add_argument("--out", required=True, help="Output file (will be written as .finezip)")
    p.add_argument("--model", default="gpt2", help="HF model repo id (default: gpt2)")
    p.add_argument("--chunk_size", type=int, default=512, help="Tokens per chunk")
    p.add_argument("--batch", type=int, default=8, help="Number of chunks to batch per forward pass")
    p.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    return p.parse_args()

def text_to_token_ids(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)

def chunk_token_ids(token_ids, chunk_size):
    lengths = []
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        c = token_ids[i:i+chunk_size]
        chunks.append(c)
        lengths.append(len(c))
    return chunks, lengths

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else args.device)

    print("Loading tokenizer & model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = text_to_token_ids(tokenizer, text)
    print(f"Tokenized {len(token_ids)} tokens.")

    chunks, chunk_lengths = chunk_token_ids(token_ids, args.chunk_size)
    print(f"Split into {len(chunks)} chunks (chunk_size={args.chunk_size}).")

    # Storing:
    # - first_tokens: first token id for each chunk (uint32)
    # - ranks: concatenated ranks (uint32) for all remaining tokens (positions 1..L-1 in each chunk)
    first_tokens = []
    all_ranks = []

    V = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else tokenizer.vocab_size
    print("Vocab size:", V)

    # Process in batches of chunks
    for i in tqdm(range(0, len(chunks), args.batch), desc="Batches"):
        batch_chunks = chunks[i:i+args.batch]
        batch_lengths = [len(c) for c in batch_chunks]
        maxlen = max(batch_lengths)

        # Prepare input tensor by padding on the right with tokenizer.eos_token_id if available, otherwise 0
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        inp = [c + [pad_id]*(maxlen - len(c)) for c in batch_chunks]
        inp_tensor = torch.tensor(inp, dtype=torch.long, device=device)

        with torch.no_grad():
            # Forward pass: for causal models, logits at position t predict token at t
            # outputs.logits shape [B, maxlen, V]
            outs = model(inp_tensor)
            logits = outs.logits  # float32 or float16 depending on model

        # For each chunk in batch process ranks for positions 1..len-1 and store first token directly
        for j, c in enumerate(batch_chunks):
            L = len(c)
            if L == 0:
                first_tokens.append(np.uint32(0))
                continue
            first_tokens.append(np.uint32(c[0]))
            # For pos = 1..L-1, the logits that predicted token at position pos are at logits[j, pos-1]
            for pos in range(1, L):
                prob_logits = logits[j, pos-1].cpu()
                true_token_id = c[pos]
                # compute rank: number of logits strictly greater than the true token logit +1
                true_logit = prob_logits[true_token_id].item()
                # Use vectorized comparison
                rank = int((prob_logits > true_logit).sum().item()) + 1
                all_ranks.append(np.uint32(rank))

    # Save metadata + arrays into an npz, then compress with bz2 to produce .finezip
    metadata = {
        "model": args.model,
        "tokenizer": args.model,
        "chunk_size": args.chunk_size,
        "num_chunks": len(chunks),
        "total_tokens": len(token_ids),
        "vocab_size": int(V),
        "chunk_lengths": np.array(chunk_lengths, dtype=np.uint32).tolist()
    }

    tmp_npz = args.out + ".tmp.npz"
    print("Saving temporary npz:", tmp_npz)
    np.savez_compressed(tmp_npz,
                        first_tokens=np.array(first_tokens, dtype=np.uint32),
                        ranks=np.array(all_ranks, dtype=np.uint32),
                        chunk_lengths=np.array(chunk_lengths, dtype=np.uint32),
                        metadata_json=json.dumps(metadata))

    # Read npz bytes and compress with bz2
    with open(tmp_npz, "rb") as f:
        raw = f.read()
    compressed = bz2.compress(raw, compresslevel=9)
    out_path = args.out if args.out.endswith(".finezip") else args.out + ".finezip"
    with open(out_path, "wb") as f:
        f.write(compressed)
    os.remove(tmp_npz)
    print("Wrote compressed file:", out_path)
    print("Done.")

if __name__ == "__main__":
    main()
