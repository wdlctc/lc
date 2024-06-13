import argparse
import os
import time
import tempfile
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from utils import load, load_jsonl, load_data
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import numpy as np

def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def main(args):
    # Specify the pretrained model name or path
    model_name = args.model_name

    # Load the tokenizer and pretrained model
    model, tokenizer = load(model_name)

    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Random data generator dataset class
    class RandomDataGenerator(Dataset):
        def __init__(self, tokenizer, num_samples, max_length):
            self.tokenizer = tokenizer
            self.num_samples = num_samples
            self.max_length = max_length
            self.vocab_size = len(tokenizer)  # Get the size of the tokenizer's vocabulary

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            random_sequence = np.random.randint(low=0, high=self.vocab_size, size=(self.max_length,))
            return torch.tensor(random_sequence)

    # Instantiate the dataset
    num_samples = args.num_samples  # Number of random samples you want to generate
    max_length = args.max_length  # Maximum length of the sequence
    dataset = RandomDataGenerator(tokenizer, num_samples, max_length)

    # DataLoader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    init_random_seed(0)
    # Inference loop
    model.eval()
    total_time = 0
    decoded_results = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch.to(device)
            start_time = time.time()
            past_key_values = None
            outputs = model(input_ids=inputs, past_key_values=past_key_values, use_cache=True)
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            print(inputs.shape, pred_token_idx.shape)
            # generated_ids = [pred_token_idx.item()]
            for _ in range(64):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            #     generated_ids.append(pred_token_idx.item())
            #     generated_text = (
            #         tokenizer.decode(
            #             generated_ids,
            #             skip_special_tokens=True,
            #             clean_up_tokenization_spaces=True,
            #             spaces_between_special_tokens=False,
            #         )
            #         .strip()
            #         .split(" ")
            #     )
                
            # print(" ".join(generated_text[:]), flush=True)
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time

    avg_inference_time = total_time / len(data_loader)
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")
    print(
        "Peak allocated bytes on {:4f}GB".format(
            torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] / 2**30
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-2-13b-hf"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="yelp_review_full"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64
    )
    parser.add_argument(
        "--num_samples", type=int, default=8
    )
    parser.add_argument(
        "--max_length", type=int, default=2264
    )
    parser.add_argument("--data_root", type=str, default="data/")
    args = parser.parse_args()
    main(args)