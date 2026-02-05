import json
import argparse
import sys
import os

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prompt import build_prompt_ids, parse_generated_label, score_candidates_logprob, predict_from_logprob


def main():
    parser = argparse.ArgumentParser(description="Few-shot evaluation on IMDB")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--samples", type=int, default=0, help="Use 0 for full test set.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--method", choices=["generate", "logprob"], default="generate",
        help="generate = decode label from generation; logprob = score labels by log-likelihood." # similar performance
    )
    parser.add_argument("--max_new_tokens", type=int, default=8, help="Only for --method generate.")
    parser.add_argument("--output", default="imdb_fewshot_results.jsonl")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print("Loading IMDB test set...")
    ds_test = load_dataset("imdb", split="test")
    if args.samples and args.samples > 0:
        ds_test = ds_test.shuffle(seed=args.seed).select(range(args.samples))
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    unknown = 0
    
    with open(args.output, "w") as f_out:
        pbar = tqdm(ds_test, desc=f"IMDB few-shot ({args.method})", unit="ex")
        
        for ex in pbar:
            gold = "positive" if int(ex["label"]) == 1 else "negative"
            prompt_ids = build_prompt_ids(tokenizer, ex["text"], max_length=args.max_length)
            attention_mask = torch.ones_like(prompt_ids)
            
            if args.method == "generate":
                prompt_ids = prompt_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)
                
                out = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                input_len = int(prompt_ids.shape[1])
                gen_ids = out[0, input_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                
                pred = parse_generated_label(gen_text)
                if pred == "unknown":
                    unknown += 1
                
                record = {
                    "id": total,
                    "gold": gold,
                    "prediction": pred,
                    "correct": pred == gold,
                    "generated_tail": gen_text,
                }
            
            else:  
                candidates = [" negative", " positive"]
                scores = score_candidates_logprob(model, tokenizer, prompt_ids, candidates)
                pred = predict_from_logprob(scores)
                
                record = {
                    "id": total,
                    "gold": gold,
                    "prediction": pred,
                    "correct": pred == gold,
                    "scores": {k.strip(): v for k, v in scores.items()},
                }
            
            total += 1
            correct += int(pred == gold)
            acc = correct / total
            
            if args.method == "generate":
                pbar.set_postfix(acc=f"{acc:.3f}", unk=unknown)
            else:
                pbar.set_postfix(acc=f"{acc:.3f}")
            
            f_out.write(json.dumps(record) + "\n")
    
    acc = correct / total if total else 0.0
    print("Few-shot Evaluation Summary")
    print(f"Model: {args.model}")
    print("Split: imdb/test")
    print(f"Method: {args.method}")
    print(f"Examples evaluated: {total}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    if args.method == "generate":
        print(f"Unknown predictions: {unknown} ({unknown/total*100:.2f}%)")
    print(f"Saved JSONL: {args.output}")


if __name__ == "__main__":
    main()