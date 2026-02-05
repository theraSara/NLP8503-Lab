import json
import argparse
import sys
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.activations import extract_activations


def main():
    parser = argparse.ArgumentParser(description="Extract activations from first sample")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--out", default="imdb_first_sample_activations.pt")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading IMDB {args.split} split...")
    ds = load_dataset("imdb", split=args.split)
    ex = ds[args.index]
    text = ex["text"]
    label = int(ex["label"])
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to(device)
    model.eval()
    
    print("Tokenizing input...")
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    print("Extracting hidden states...")
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = tuple(h.detach().cpu() for h in out.hidden_states)
    
    print("Extracting MLP activations...")
    mlp_acts = extract_activations(model, tokenizer, input_ids, attention_mask)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    decoded_text = tokenizer.decode(input_ids[0].detach().cpu(), skip_special_tokens=True)
    
    payload = {
        "meta": {
            "model": args.model,
            "dataset": "imdb",
            "split": args.split,
            "index": args.index,
            "label": label,
            "max_length": args.max_length,
            "device": device,
            "dtype": str(dtype),
            "seq_len": int(input_ids.shape[1]),
            "num_hidden_state_tensors": len(hidden_states),
            "num_mlp_layers_captured": len(mlp_acts),
        },
        "tokens": tokens,
        "decoded_text_seen": decoded_text,
        "hidden_states": hidden_states,
        "mlp_activations": mlp_acts,
    }
    
    torch.save(payload, args.out)
    print(f"Saved activations to: {args.out}")
    print(f"Metadata:")
    print(json.dumps(payload["meta"], indent=2))


if __name__ == "__main__":
    main()