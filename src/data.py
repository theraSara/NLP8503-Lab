import torch
import numpy as np
from datasets import load_dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_balanced(dataset_name, split, samples, seed=0):
    ds = load_dataset(dataset_name, split=split).shuffle(seed=seed)

    all_samples = []
    counts = {}

    for ex in ds:
        label = int(ex["label"])
        if label not in counts:
            counts[label]=0

        if counts[label] < samples:
            all_samples.append(ex)
            counts[label] += 1

        if all(c >= samples for c in counts.values()):
            break

    return all_samples

def fewshot_prompt(review_text, header, examples):
    review_text = review_text.strip().replace("\n", " ")
    prompt = header + examples + f"Review: {review_text}\nSentiment:"
    return prompt