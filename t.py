import torch

from accelerate import init_empty_weights

model_name = "configs/llama3_8b.json"

import torch
import json
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load(model_name):
    print(f"Loading model from {model_name} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=8192)

    config = AutoConfig.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_config(
        config,
        # device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0


    return model, tokenizer

# Load the tokenizer and pretrained model
model, tokenizer = load(model_name)

pad_idx = tokenizer.pad_token_id

# Move the model to GPU(s)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)