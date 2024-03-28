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

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

from rtp.rotated_tensor_parallel import RotatedTensorParallel

RPC_PORT = 29501

def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)



def benchmark_dp(rank, args, world_size):
    """Benchmark a given model using a single process and multiple devices."""
    init_method_pgroup = "tcp://localhost:{}".format(RPC_PORT)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method=init_method_pgroup
    )

    torch.cuda.set_device(rank)
    init_random_seed(0)
    
    # Specify the pretrained model name or path
    model_name = args.model_name
    
    # Load the tokenizer and pretrained model
    model, tokenizer = load(model_name)
    
    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model = RotatedTensorParallel(model, inplace=True)

    model.train()
    
    print(model)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
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
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
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
            generated_ids = [pred_token_idx.item()]
            for _ in range(50):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_ids.append(pred_token_idx.item())
                generated_text = (
                    tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    )
                    .strip()
                    .split(" ")
                )
                
            print(" ".join(generated_text[:]), flush=True)
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time

    avg_inference_time = total_time / len(data_loader)
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")
    print(
        "Peak allocated bytes on {:4f}GB".format(
            torch.cuda.memory_stats(rank)["allocated_bytes.all.peak"] / 2**30
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="openai-community/gpt2"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="yelp_review_full"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--num_samples", type=int, default=10
    )
    parser.add_argument(
        "--max_length", type=int, default=512
    )
    parser.add_argument("--data_root", type=str, default="data/")
    args = parser.parse_args()
    
    print(f"Running DP benchmark with args: {args}")
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(torch.cuda.device_count())

    mp.spawn(
        benchmark_dp,
        args=(args, num_devices),
        nprocs=num_devices,
        join=True,
    )