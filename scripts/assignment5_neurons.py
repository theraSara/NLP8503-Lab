
import os
import json
import argparse
import sys

import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.activations import extract_activations
from src.data import load_balanced, set_seed
from src.prompt import build_prompt_ids, get_decision_positions


def offset_abs(seq_len, pos_offset):
    if pos_offset < 0:
        return seq_len + pos_offset
    return pos_offset

def feature_layout(model, tokenizer, sample_text, max_length, capture, pos_offset):
    prompt_ids = build_prompt_ids(tokenizer, sample_text, max_length=max_length)
    seq_len = int(prompt_ids.shape[1])
    abs_pos = offset_abs(seq_len, pos_offset)

    if abs_pos < 0 or abs_pos >= seq_len:
        raise ValueError(f"pos_offset={pos_offset} out of range for seq_len={seq_len}")
    
    input_ids = prompt_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    acts = extract_activations(
        model,
        tokenizer,
        input_ids,
        attention_mask,
        capture=capture,
        positions=[abs_pos]
    )

    layers = sorted(acts.keys())
    per_layer_dim = [int(acts[l].shape[2]) for l in layers]
    offsets = np.cumsum([0]+per_layer_dim).astype(int)
    total_dim = int(offsets[-1])

    meta = {
        "layers": layers,
        "per_layer_dim": per_layer_dim,
        "offsets": offsets.tolist(),
        "total_dim": total_dim,
        "pos_offset": int(pos_offset),
        "example_seq_len": seq_len,
        "example_abs_pos": int(abs_pos),
        "capture": capture,
    }

    return meta


def build_activation_matrix(model, tokenizer, samples, max_length, capture, pos_offset, meta):

    layers = meta["layers"]
    per_layer_dim = meta["per_layer_dim"]
    offsets = np.array(meta["offsets"], dtype=int)
    total_dim = int(meta["total_dim"])

    rows = []
    labels = []
    kept_indices = []

    for i, ex in enumerate(tqdm(samples, desc="Extracting activations")):
        prompt_ids = build_prompt_ids(tokenizer, ex["text"], max_length=max_length)
        seq_len = int(prompt_ids.shape[1])
        abs_pos = offset_abs(seq_len, pos_offset)

        if abs_pos < 0 or abs_pos >= seq_len:
            continue

        input_ids = prompt_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)

        acts = extract_activations(
            model, 
            tokenizer, 
            input_ids, 
            attention_mask, 
            capture=capture, 
            positions=[abs_pos]
        )

        row = np.zeros(total_dim, dtype=np.float32)
        for li, layer in enumerate(layers):
            D = per_layer_dim[li]
            vec = acts[layer][0,0].numpy().astype(np.float32)
            row[offsets[li]:offsets[li+1]] = vec

        rows.append(row)
        labels.append(int(ex["label"]))
        kept_indices.append(i)

    X = np.stack(rows, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    return X, y, kept_indices

def rank_top_neurons(coef, X_train, topk, method="std"):
    """
    method:
      - "abs": importance = |w|
      - "std": importance = |w| * std(feature)
    """
    w = coef.reshape(-1)
    abs_w = np.abs(w)

    if method == "abs":
        importance = abs_w
    elif method == "std":
        feat_std = X_train.std(axis=0).astype(np.float32)
        importance = abs_w * (feat_std + 1e-8)
    else:
        raise ValueError("selected method must be 'abs' or 'std'")

    idx_sorted = np.argsort(-importance)
    top_idx = idx_sorted[:topk]
    return top_idx, w[top_idx], importance[top_idx]


def global_layer_map(meta, global_idx):
    offsets = np.array(meta["offsets"], dtype=int)
    layers = meta["layers"]

    li = int(np.searchsorted(offsets, global_idx, side="right") - 1)
    local = int(global_idx - offsets[li])
    layer = int(layers[li])
    return layer, local


def main():
    parser = argparse.ArgumentParser(description="Assignment 5: Identify top predictive neurons")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--train_per_class", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--pos_offset", type=int, default=-1)
    parser.add_argument("--capture", choices=["down_proj_in", "gate_act"], default="down_proj_in")

    # Probe training
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--penalty", choices=["l2", "l1"], default="l2") # Use l1 for sparsity (more interpretable), l2 for stable dense weights.
    parser.add_argument("--max_iter", type=int, default=2000)

    # Ranking
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--rank_method", choices=["abs", "std"], default="std")
    parser.add_argument("--out_prefix", default="top_neurons")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    
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
    
    print("Loading training data...")
    train_samples = load_balanced("imdb", "train", args.train_per_class, args.seed) # balanced data
    print(f"Loaded {len(train_samples)} training samples")

    meta = feature_layout(
        model=model,
        tokenizer=tokenizer,
        sample_text=train_samples[0]["text"],
        max_length=args.max_length,
        capture=args.capture,
        pos_offset=args.pos_offset
    )
    total_dim = meta["total_dim"]
    print(f"Feature dimension: {total_dim} (layer={len(meta['layers'])}, capture={args.capture}, pos_offset={args.pos_offset})")

    
    X_train, y_train, kept = build_activation_matrix(
        model=model, 
        tokenizer=tokenizer, 
        samples=train_samples, 
        max_length=args.max_length,
        capture=args.capture,
        pos_offset=args.pos_offset,
        meta=meta
    )
    print("X_train shape: ", X_train.shape)
    
    print("Training logistic regression...")
    clf = LogisticRegression(
        C=args.C,
        penalty=args.penalty,
        solver="saga", 
        max_iter=args.max_iter, 
        n_jobs=1,
        random_state=args.seed,
        verbose=0
    )
    clf.fit(X_train, y_train)
    print("Training complete")
    
    coef = clf.coef_.reshape(-1)

    top_idx, top_w, top_imp = rank_top_neurons(
        coef=coef,
        X_train=X_train,
        topk=args.topk,
        method=args.rank_method
    )

    top_neurons = []
    layer_counts = {int(l): 0 for l in meta["layers"]}
    
    for gi, w_val, imp_val in zip(top_idx.tolist(), top_w.tolist(), top_imp.tolist()):
        layer, local = global_layer_map(meta, gi)        
        layer_counts[layer] += 1
        top_neurons.append({
            "global_index": int(gi),
            "layer": int(layer),
            "layer_local_index": int(local),
            "coef": float(w_val),
            "importance": float(imp_val)
        })
    
    out_json = f"{args.out_prefix}.json"
    with open(out_json, "w") as f:
        json.dump({
            "meta": {
                **meta,
                "model": args.model,
                "dataset": "imdb",
                "split": "train",
                "train_per_class": args.train_per_class,
                "max_length": args.max_length,
                "C": args.C,
                "penalty": args.penalty,
                "rank_method": args.rank_method,
                "topk": args.topk,
                "seed": args.seed
            },
            "layer_counts": layer_counts,
            "top_neurons": top_neurons,
        }, f, indent=2)
    print(f"Saved: {out_json}")
    
    out_csv = f"{args.out_prefix}_hist.csv"
    with open(out_csv, "w") as f:
        f.write("layer,count\n")
        for layer in sorted(layer_counts.keys()):
            f.write(f"{layer},{layer_counts[layer]}\n")
    print(f"Saved: {out_csv}")

    np.save(f"{args.out_prefix}_idx.npy", top_idx)
    print(f"Saved: {args.out_prefix}_idx.npy")

    sorted_counts = sorted(layer_counts.items(), key=lambda kv: -kv[1])
    print("\nTop layers by number of top neurons:")
    for layer, count in sorted_counts[:10]:
        print(f"  Layer {layer}: {count}")


if __name__ == "__main__":
    main()