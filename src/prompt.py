import re
import torch

HEADER = (
    "Task: Classify the sentiment of the movie review as positive or negative.\n"
    "Answer with exactly one word: positive or negative.\n\n"
)

EXAMPLES = (
    "Review: I hated this movie. It was boring and pointless.\n"
    "Sentiment: negative\n\n"
    "Review: A wonderful film with great acting. I loved it.\n"
    "Sentiment: positive\n\n"
)

LABEL_RE = re.compile(r"^(positive|negative)\b", re.IGNORECASE)


def normalize_review(text):
    return text.strip().replace("\n", " ")


def build_prompt_text(review_text):
    review_text = normalize_review(review_text)
    return HEADER + EXAMPLES + f"Review: {review_text}\nSentiment:"


def build_prompt_ids(tokenizer, review_text, max_length=2048):
    prefix = HEADER + EXAMPLES + "Review: "
    suffix = "\nSentiment:"
    
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    review_ids = tokenizer(normalize_review(review_text), add_special_tokens=False).input_ids
    
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    
    max_review = max_length - len(bos) - len(prefix_ids) - len(suffix_ids)
    if max_review < 0:
        raise ValueError(
            f"max_length={max_length} too small. Needs at least {len(bos) + len(prefix_ids) + len(suffix_ids)} tokens to fit the prompt."
        )
    
    if len(review_ids) > max_review:
        review_ids = review_ids[:max_review]
    
    input_ids = bos + prefix_ids + review_ids + suffix_ids
    return torch.tensor([input_ids], dtype=torch.long)


def get_suffix_len(tokenizer):
    suffix = "\nSentiment:"
    suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    return len(suffix_ids)


def get_decision_positions(tokenizer, review_text, max_length=2048, offsets=None):
    if offsets is None:
        offsets = [-1, -2]
    
    prompt_ids = build_prompt_ids(tokenizer, review_text, max_length)
    seq_len = int(prompt_ids.shape[1])
    
    abs_positions = []
    for offset in offsets:
        if offset < 0:
            pos = seq_len + offset
        else:
            pos = offset
        
        if pos < 0 or pos >= seq_len:
            return None
        
        abs_positions.append(pos)
    
    return abs_positions


def parse_generated_label(generated_text):
    s = generated_text.strip().lower()
    
    m = LABEL_RE.match(s)
    if m:
        return m.group(1).lower()
    
    early = s[:50]
    if "positive" in early and "negative" not in early:
        return "positive"
    if "negative" in early and "positive" not in early:
        return "negative"
    
    return "unknown"


@torch.inference_mode()
def score_candidates_logprob(model, tokenizer, prompt_ids, candidates):
    device = model.device
    prompt_ids = prompt_ids.to(device)
    prompt_len = int(prompt_ids.shape[1])
    
    cand_ids_list = []
    for c in candidates:
        c_ids = tokenizer(c, add_special_tokens=False).input_ids
        cand_ids_list.append(torch.tensor([c_ids], dtype=torch.long, device=device))
    
    full_list = [torch.cat([prompt_ids, cids], dim=1) for cids in cand_ids_list]
    max_len = max(int(x.shape[1]) for x in full_list)
    bsz = len(full_list)
    
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    
    input_batch = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
    attn_batch = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    
    for i, x in enumerate(full_list):
        L = int(x.shape[1])
        input_batch[i, :L] = x[0]
        attn_batch[i, :L] = 1
    
    logits = model(input_batch, attention_mask=attn_batch, use_cache=False).logits
    logp = torch.log_softmax(logits, dim=-1)
    
    scores = {}
    for i, (cand, cids) in enumerate(zip(candidates, cand_ids_list)):
        clen = int(cids.shape[1])
        total = 0.0
        for j in range(clen):
            tok = int(cids[0, j].item())
            total += float(logp[i, prompt_len + j - 1, tok].item())
        scores[cand] = total
    
    return scores


def predict_from_logprob(scores):
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best.strip().lower()