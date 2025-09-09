# run_model.py
# Generate completions for both train and test set

import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface_cache"
import torch

from utils.utils import generate_formatted_prompts, generate, save_completions
from transformers import AutoTokenizer, AutoModelForCausalLM



def main():
    creative_system = (
        "You are an assistant who uses creative language."
    )
    formal_system = (
        "You are an assistant who uses formal language."
    )
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = "auto",
        torch_dtype=torch.float16,  # Use fp16 to cut memory usage
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    for data_split in ["train", "test", "val"]:
        # Load prompts
        creative_prompts, formal_prompts = generate_formatted_prompts(creative_system, formal_system, f"prompts/{data_split}_prompts.txt")

        formal_dest_file = os.path.join("completions", data_split, "formal.txt")
        creative_dest_file = os.path.join("completions", data_split, "creative.txt")

        save_completions(creative_prompts, model, tokenizer, creative_dest_file)
        save_completions(formal_prompts, model, tokenizer, formal_dest_file)


if __name__ == "__main__":
    main()
