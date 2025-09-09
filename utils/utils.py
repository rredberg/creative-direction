import torch
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface_cache"

import random
from tqdm import tqdm


# -------------------------------
# Data loading, generation, prompt formatting, and saving outputs
# -------------------------------

DELIM = "\n\n-------------------\n\n"

# -------------------------------
# Prompt Handling
# -------------------------------

def load_prompts_from_file(path):
    with open(path, "r") as f:
        text = f.read()
    prompts = [p.strip() for p in text.split(DELIM) if p.strip()]
    return prompts

def split_prompts(
    input_file='prompts/prompts.txt',
    train_file='prompts/train_prompts.txt',
    val_file='prompts/val_prompts.txt',
    test_file='prompts/test_prompts.txt',
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
):
    """
    Splits prompts.txt into train, val, and test files.
    Default split: 70% train, 15% val, 15% test.
    """

    # Load and clean prompts
    with open(input_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(prompts)

    total = len(prompts)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_prompts = prompts[:train_end]
    val_prompts = prompts[train_end:val_end]
    test_prompts = prompts[val_end:]

    # Write to files
    with open(train_file, 'w', encoding='utf-8') as f:
        for p in train_prompts:
            f.write(p + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for p in val_prompts:
            f.write(p + '\n')

    with open(test_file, 'w', encoding='utf-8') as f:
        for p in test_prompts:
            f.write(p + '\n')

    # Print stats
    print(f"Total prompts: {total}")
    print(f"Train prompts: {len(train_prompts)}")
    print(f"Validation prompts: {len(val_prompts)}")
    print(f"Test prompts: {len(test_prompts)}")


def generate_formatted_prompts(creative_system, formal_system, prompts_file):
    with open(prompts_file, "r") as f:
        prompts = [p.strip() for p in f.readlines() if p.strip()]
    creative_prompts = [f"{creative_system}\nUser: {p}\nAssistant:" for p in prompts]
    formal_prompts = [f"{formal_system}\nUser: {p}\nAssistant:" for p in prompts]
    return creative_prompts, formal_prompts

# -------------------------------
# Text Generation / Clean-up
# -------------------------------

def generate(model, prompt, tokenizer, max_new_tokens=300):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def truncate_after_first_user_followup(full_text):
    # Split by "User:" to find user turns
    user_splits = full_text.split("User:")

    if len(user_splits) <= 2:
        # No follow-up user turn, keep full
        return full_text.strip()

    # Keep system message (if any), first user turn, and assistant response
    # Reconstruct text: everything before second "User:"
    truncated = "User:".join(user_splits[:2]).strip()
    return truncated

def save_completions(prompts, model, tokenizer, dest_file, min_token_length=20):
    """
    Generates and saves completions for the given prompts,
    skipping ones where the assistant response is too short (after truncation).
    """
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)

    for prompt in prompts:
        # Generate full completion (includes prompt + model output)
        completion = generate(model, prompt, tokenizer)

        # Truncate any second "User:" turn if it exists
        completion = truncate_after_first_user_followup(completion)

        # Save the full truncated completion
        with open(dest_file, "a", encoding="utf-8") as f:
            f.write(completion + DELIM)

        print(f"\n---\n{completion}\n---\n")


# -------------------------------
# Activations / Hook Stuff 
# -------------------------------

def get_hook(layer_idx, activations):
    def hook_fn(module, input, output):
        activations[layer_idx] = input[0].detach().cpu()
    return hook_fn


def save_all_token_activations(prompt_list, model, tokenizer, save_dir, activations):
    os.makedirs(save_dir, exist_ok=True)

    for idx, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
        activations.clear()  # Clear previous run

        # Tokenize full prompt and send to model device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)

        # Check if "Assistant:" exists in prompt
        if "Assistant:" not in prompt:
            print(f"Skipping index {idx} — no 'Assistant:' found")
            continue

        # Find position where assistant response starts
        assistant_start = prompt.find("Assistant:")
        prompt_prefix = prompt[:assistant_start + len("Assistant:")]

        # Tokenize prompt_prefix to find index of assistant start tokens
        prefix_ids = tokenizer(prompt_prefix, return_tensors="pt")["input_ids"]
        prompt_len = prefix_ids.size(1)  # sequence length of prefix tokens

        # Slice activations to only include tokens from assistant response onward
        activation_summary = {
            layer_idx: tensor[0, prompt_len:]  # shape: (num_response_tokens, hidden_dim)
            for layer_idx, tensor in activations.items()
        }

        # Save activation dictionary as a .pt file
        torch.save(activation_summary, os.path.join(save_dir, f"prompt_{idx}.pt"))

