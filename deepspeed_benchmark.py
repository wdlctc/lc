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
import deepspeed

RPC_PORT = 29505

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
    num_samples = 100  # Number of random samples you want to generate
    max_length = 512  # Maximum length of the sequence
    dataset = RandomDataGenerator(tokenizer, num_samples, max_length)

    batch_size = 2
    # DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize DeepSpeed
    deepspeed_config = {
        "train_batch_size": batch_size,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 5e-5
            }
        },
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
        },
        "dump_state": False,
        "steps_per_print": 1000000,
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=None,
        model=model,
        model_parameters=model.parameters(),
        config=deepspeed_config
    )
    
    # Set up the optimizer
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
    
        for batch in data_loader:
            inputs = batch.to(device)
            outputs = model_engine(input_ids=inputs, labels=inputs)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            total_loss += loss.item()
    
        avg_loss = total_loss / len(data_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f} - Time: {epoch_time:.2f} seconds")


    print(
        "Peak allocated bytes on cuda:{}: {:4f}GB".format(
            dist.get_rank(), torch.cuda.memory_stats(dist.get_rank())["allocated_bytes.all.peak"] / 2**30
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="google-bert/bert-large-uncased"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="yelp_review_full"
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--data_root", type=str, default="data/")
    args = parser.parse_args()
    
    print(f"Running DP benchmark with args: {args}")
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(torch.cuda.device_count())

    benchmark_dp(args.local_rank, args, num_devices)