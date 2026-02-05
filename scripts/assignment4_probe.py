import csv
import argparse
import sys
import os

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activations import extract_activations
from src.data import load_balanced
from src.prompt import build_prompt_ids, get_decision_positions
from src.probe import train_linear_probe


def layer_features(model, tokenizer, samples, pos_offset, max_length, capture):
    device = next(model.parameters()).device
    num_layers = len(model.model.layers)

    X_lists = {layer: [] for layer in range(num_layers)}
    y_list = []

    for ex in tqdm(samples, desc=f"Extracting features (pos_offset={pos_offset})"):
        prompt_ids = build_prompt_ids(tokenizer, ex["text"], max_length=max_length)
        abs_positions = get_decision_positions(
            tokenizer,
            ex["text"],
            max_length=max_length,
            offsets=[pos_offset]
        )
        if abs_positions is None:
            continue

        input_ids = prompt_ids.to(device)
        attention_mask = torch.ones_like(input_ids, device=device)

        acts = extract_activations(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            capture=capture,
            positions=abs_positions
        )

        for layer in range(num_layers):
            vec = acts[layer][0, 0].numpy() 
            X_lists[layer].append(vec)

        y_list.append(int(ex["label"]))

    y = np.array(y_list, dtype=np.int64)
    X_by_layer = {}
    for layer, lst in X_lists.items():
        X_by_layer[layer] = np.stack(lst, axis=0).astype(np.float32)

    return X_by_layer, y


def main():
    parser = argparse.ArgumentParser(description="Assignment 4: Linear probing across layers")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--train_per_class", type=int, default=500)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos_offsets", nargs="+", type=int, default=[-3, -2, -1])
    parser.add_argument("--capture", choices=["gate_act", "down_proj_in"], default="down_proj_in")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1500)
    parser.add_argument("--no_scale", action="store_true")
    parser.add_argument("--out_prefix", default="imdb_linear_probe")
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

    print("Loading balanced training data...")
    train_samples = load_balanced("imdb", "train", args.train_per_class, args.seed)
    print(f"Train samples: {len(train_samples)}")

    print("Loading test data...")
    ds_test = load_dataset("imdb", split="test").shuffle(seed=args.seed)
    test_samples = list(ds_test.select(range(args.test_samples)))
    print(f"Test samples: {len(test_samples)}")

    num_layers = len(model.model.layers)

    for pos_offset in args.pos_offsets:
        print("\n" + "=" * 70)
        print(f"Probing for pos_offset={pos_offset} (capture={args.capture})")

        X_train_by_layer, y_train = layer_features(
            model, tokenizer, train_samples, pos_offset, args.max_length, args.capture
        )
        X_test_by_layer, y_test = layer_features(
            model, tokenizer, test_samples, pos_offset, args.max_length, args.capture
        )

        layer_acc = {}
        for layer in range(num_layers):
            X_train = X_train_by_layer[layer]
            X_test = X_test_by_layer[layer]

            clf, acc = train_linear_probe(
                X_train, y_train,
                X_test, y_test,
                C=args.C,
                max_iter=args.max_iter,
                seed=args.seed,
                scale=(not args.no_scale)
            )
            layer_acc[layer] = acc

        out_csv = f"{args.out_prefix}_pos{pos_offset}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pos_offset", "layer", "accuracy"])
            for layer in range(num_layers):
                w.writerow([pos_offset, layer, layer_acc[layer]])

        best_layer = max(layer_acc, key=layer_acc.get)
        print(f"Saved: {out_csv}")
        print(f"Best layer @ pos_offset={pos_offset}: {best_layer} (acc={layer_acc[best_layer]:.4f})")


if __name__ == "__main__":
    main()
