import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface_cache"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from utils.utils import get_hook, save_all_token_activations, load_prompts_from_file



def main():
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    
    activations = {}
    
    # Register hooks on all layers
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(get_hook(idx, activations)))
    
    # Load prompts from  completions folder
    data_paths = {
        "train/creative": "completions/train/creative.txt",
        "train/neutral": "completions/train/neutral.txt",
        "test/creative": "completions/test/creative.txt",
        "test/neutral": "completions/test/neutral.txt",
        "val/creative": "completions/val/creative.txt",
        "val/neutral": "completions/val/neutral.txt",
    }
    
    for key, path in data_paths.items():
        prompts = load_prompts_from_file(path)
        save_dir = f"activations/{key}"
        save_all_token_activations(prompts, model, tokenizer, save_dir, activations)
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()



if __name__ == "__main__":
    main()
