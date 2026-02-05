import sys
import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt


def ablate_neurons_hook(layer_idx, neuron_indices_to_ablate):
    def hook(module, inp, out):
        out = out.clone()
        out[:, :, neuron_indices_to_ablate] = 0.0
        return out
    return hook


def evaluate_with_ablation(model, tokenizer, test_samples, neurons_to_ablate,  max_length=2048, device="cuda"):
    handles = []
    layers = model.model.layers
    
    for layer_idx, neuron_idxs in neurons_to_ablate.items():
        if len(neuron_idxs) > 0:
            hook = ablate_neurons_hook(layer_idx, neuron_idxs)
            h = layers[layer_idx].mlp.gate_proj.register_forward_hook(hook)
            handles.append(h)
    
    correct = 0
    total = 0
    
    model.eval()
    for ex in tqdm(test_samples, desc="Evaluating with ablation", leave=False):
        prompt = create_fewshot_prompt(ex["text"])
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = decoded[len(prompt):].strip().lower()
        
        if answer.startswith("positive"):
            pred = 1
        elif answer.startswith("negative"):
            pred = 0
        else:
            pred = 0
        
        gold = int(ex["label"])
        if pred == gold:
            correct += 1
        total += 1
    
    for h in handles:
        h.remove()
    
    return correct / total if total > 0 else 0.0


def create_fewshot_prompt(review_text):
    header = (
        "Task: Classify the sentiment of the movie review as positive or negative.\n"
        "Answer with exactly one word: positive or negative.\n\n"
    )
    examples = (
        "Review: I hated this movie. It was boring and pointless.\n"
        "Sentiment: negative\n\n"
        "Review: A wonderful film with great acting. I loved it.\n"
        "Sentiment: positive\n\n"
    )
    review_text = review_text.strip().replace("\n", " ")
    return header + examples + f"Review: {review_text}\nSentiment:"


def main():
    parser = argparse.ArgumentParser(description="Ablation study on top neurons")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--top_neurons_file", default="top_neurons.json",
                        help="JSON file with top neurons from Assignment 5")
    parser.add_argument("--test_samples", type=int, default=500,
                        help="Number of test samples to evaluate")
    parser.add_argument("--ablation_steps", nargs="+", type=int, default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        help="Numbers of neurons to ablate")
    parser.add_argument("--n_random_trials", type=int, default=3,
                        help="Number of random ablation trials to average")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--output_dir", default="results/ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device != "cuda":
        model.to(device)
    model.eval()
    
    print("Loading test data...")
    ds_test = load_dataset("imdb", split="test").shuffle(seed=args.seed)
    test_samples = list(ds_test.select(range(args.test_samples)))
    
    print(f"Loading top neurons from {args.top_neurons_file}...")
    with open(args.top_neurons_file, "r") as f:
        top_data = json.load(f)
    
    top_neurons = top_data["top_neurons"]
    
    neurons_by_layer = {}
    for neuron in top_neurons:
        layer = neuron["layer"]
        if layer not in neurons_by_layer:
            neurons_by_layer[layer] = []
        neurons_by_layer[layer].append(neuron)
    
    print("Evaluating baseline (no ablation)...")
    baseline_acc = evaluate_with_ablation(
        model, tokenizer, test_samples, {}, args.max_length, device
    )
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    results_top = [(0, baseline_acc)]
    results_random = [(0, baseline_acc)]
    
    num_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size
    
    for n_ablate in args.ablation_steps:
        if n_ablate > len(top_neurons):
            print(f"Skipping {n_ablate} (more than total top neurons)")
            continue
        
        print(f"Ablating {n_ablate} neurons...")
        
        print("Ablating top neurons...")
        top_n = top_neurons[:n_ablate]
        
        ablation_dict_top = {}
        for neuron in top_n:
            layer = neuron["layer"]
            idx = neuron["layer_local_index"]
            if layer not in ablation_dict_top:
                ablation_dict_top[layer] = []
            ablation_dict_top[layer].append(idx)
        
        acc_top = evaluate_with_ablation(
            model, tokenizer, test_samples, ablation_dict_top, 
            args.max_length, device
        )
        results_top.append((n_ablate, acc_top))
        print(f"Top neurons removed: {n_ablate}, Accuracy: {acc_top:.4f}")
        
        print(f"Ablating random neurons ({args.n_random_trials} trials)...")
        random_accs = []
        
        for trial in range(args.n_random_trials):
            np.random.seed(args.seed + trial)
            
            random_neurons = []
            while len(random_neurons) < n_ablate:
                layer = np.random.randint(0, num_layers)
                neuron_idx = np.random.randint(0, intermediate_size)
                
                if (layer, neuron_idx) not in [(n["layer"], n["layer_local_index"]) 
                                                for n in random_neurons]:
                    random_neurons.append({
                        "layer": layer,
                        "layer_local_index": neuron_idx
                    })
            
            ablation_dict_random = {}
            for neuron in random_neurons:
                layer = neuron["layer"]
                idx = neuron["layer_local_index"]
                if layer not in ablation_dict_random:
                    ablation_dict_random[layer] = []
                ablation_dict_random[layer].append(idx)
            
            acc_random = evaluate_with_ablation(
                model, tokenizer, test_samples, ablation_dict_random,
                args.max_length, device
            )
            random_accs.append(acc_random)
            print(f"  Trial {trial+1}: {acc_random:.4f}")
        
        avg_random_acc = np.mean(random_accs)
        results_random.append((n_ablate, avg_random_acc))
        print(f"Random neurons removed: {n_ablate}, Avg Accuracy: {avg_random_acc:.4f}")
    
    results_file = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "baseline_accuracy": baseline_acc,
            "top_neurons_ablation": results_top,
            "random_neurons_ablation": results_random,
            "config": vars(args)
        }, f, indent=2)
    print(f"\nSaved results to {results_file}")
    
    plot_ablation_results(results_top, results_random, baseline_acc, args.output_dir)
    
    print("ABLATION STUDY SUMMARY")
    print(f"Baseline accuracy: {baseline_acc:.4f}\n")
    
    print("Performance drop (top neurons):")
    for n, acc in results_top[1:]:
        drop = baseline_acc - acc
        print(f"  {n:4d} neurons: {acc:.4f} (drop: {drop:.4f})")
    
    print("\nPerformance drop (random neurons):")
    for n, acc in results_random[1:]:
        drop = baseline_acc - acc
        print(f"  {n:4d} neurons: {acc:.4f} (drop: {drop:.4f})")


def plot_ablation_results(results_top, results_random, baseline_acc, output_dir):
    n_removed_top, acc_top = zip(*results_top)
    n_removed_rand, acc_rand = zip(*results_random)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_removed_top, acc_top, marker='o', linewidth=2, label='Remove Top Neurons', color='red', markersize=8)
    plt.plot(n_removed_rand, acc_rand, marker='s', linewidth=2, label='Remove Random Neurons', color='blue', markersize=8)
    plt.axhline(baseline_acc, color='green', linestyle='--', label=f'Baseline ({baseline_acc:.3f})', linewidth=2)
    
    plt.xlabel("Number of Neurons Removed", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title("Ablation Study: Top vs Random Neurons (IMDB Sentiment)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "ablation_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()