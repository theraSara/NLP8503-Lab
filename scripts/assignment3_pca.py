
import argparse
import sys
import os

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.decomposition import IncrementalPCA
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.activations import extract_activations
from src.prompt import build_prompt_ids

def offsets_positions(pos_offsets, seq_len):
    abs_pos = []
    for off in pos_offsets:
        idx = seq_len + off if off < 0 else off
        if idx < 0 or idx >= seq_len:
            return None
        abs_pos.append(idx)
    return abs_pos

def main():
    parser = argparse.ArgumentParser(description="Assignment 3: PCA analysis of MLP activations")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--pos_offsets", nargs="+", type=int, default=[-3, -2, -1])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--capture", choices=["gate_act", "down_proj_in"], default="down_proj_in")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", default="imdb_PCA_all.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    num_layers = len(model.model.layers)
    layers = list(range(num_layers))
    pos_offsets = args.pos_offsets
    
    print(f"Loading IMDB test set: {args.samples} samples (seed={args.seed})")
    ds_test = load_dataset("imdb", split="test").shuffle(seed=args.seed).select(range(args.samples))
    
    labels = torch.tensor([int(ex["label"]) for ex in ds_test], dtype=torch.int64)

    ipca = {}
    buffers = {}
    for layer in layers:
        for off in pos_offsets:
            ipca[(layer, off)] = IncrementalPCA(n_components=2, batch_size=args.batch_size)
            buffers[(layer, off)] = []
    
    print("Fitting PCA models...")
    for idx, ex in enumerate(tqdm(ds_test, desc="PCA fit")):
        prompt_ids = build_prompt_ids(tokenizer, ex["text"], max_length=args.max_length)
        seq_len = int(prompt_ids.shape[1])
        abs_positions = offsets_positions(pos_offsets, seq_len)
        if abs_positions is None:
            continue

        input_ids = prompt_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)

        # extract only the positions
        acts = extract_activations(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            capture=args.capture,
            positions=abs_positions
        )

        for layer in layers:
            A = acts[layer][0].numpy().astype(np.float32)
            for j, off in enumerate(pos_offsets):
                buffers[(layer, off)].append(A[j])

        if (idx + 1) % args.batch_size == 0:
            for key, buf in buffers.items():
                if len(buf) >= 2:
                    Xb = np.stack(buf, axis=0)
                    ipca[key].partial_fit(Xb)
                buffers[key] = []

    for key, buf in buffers.items():
        if len(buf) >= 2:
            Xb = np.stack(buf, axis=0)
            ipca[key].partial_fit(Xb)
        buffers[key] = []


    print("Transforming to 2D...")
    coords_2d = {
        layer: {off: torch.empty((args.samples, 2), dtype=torch.float32) for off in pos_offsets}
        for layer in layers
    }

    for i, ex in enumerate(tqdm(ds_test, desc="PCA transform")):
        prompt_ids = build_prompt_ids(tokenizer, ex["text"], max_length=args.max_length)
        seq_len = int(prompt_ids.shape[1])
        abs_positions = offsets_positions(pos_offsets, seq_len)
        if abs_positions is None:
            for layer in layers:
                for off in pos_offsets:
                    coords_2d[layer][off][i] = torch.tensor([float("nan"), float("nan")])
            continue

        input_ids = prompt_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)

        acts = extract_activations(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            capture=args.capture,
            positions=abs_positions
        )

        for layer in layers:
            A = acts[layer][0].numpy().astype(np.float32)
            for j, off in enumerate(pos_offsets):
                z = ipca[(layer, off)].transform(A[j:j+1])[0]
                coords_2d[layer][off][i] = torch.tensor(z, dtype=torch.float32)

    explained_variance_ratio = {
        layer: {off: ipca[(layer, off)].explained_variance_ratio_.tolist() for off in pos_offsets}
        for layer in layers
    }

    payload = {
        "meta": {
            "model": args.model,
            "dataset": "imdb",
            "split": "test",
            "samples": args.samples,
            "seed": args.seed,
            "max_length": args.max_length,
            "capture": args.capture,
            "layers": layers,
            "pos_offsets": pos_offsets,
        },
        "labels": labels,
        "coords_2d": coords_2d,
        "explained_variance_ratio": explained_variance_ratio
    }

    torch.save(payload, args.out)
    print(f"\nSaved PCA file to: {args.out}")
    print("Layers:", len(layers), "Positions:", pos_offsets)



if __name__ == "__main__":
    main()
