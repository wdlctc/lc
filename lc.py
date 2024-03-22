import argparse
import os

import torch

from utils import load, load_jsonl

def inference(model, tokenizer, prompts, max_length=50):
    for idx, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            output = model.generate(input_ids, max_length=100, num_return_sequences=1)
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        print("Generated text:", generated_text)

def main(args):
    # Specify the pretrained model name or path
    model_name = args.model_name
    
    # Load the tokenizer and pretrained model
    model, tokenizer = load(model_name)
    
    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")
    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    inference(model, tokenizer, prompts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    args = parser.parse_args()

    main(args)