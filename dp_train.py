import argparse
import os
import time

import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

from torch.optim import AdamW

from utils import load, load_jsonl, load_data
from datasets import load_dataset, load_from_disk

from transformers import TrainingArguments, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM

import numpy as np
import datasets
from torch.utils.data import IterableDataset, DataLoader

import torch.multiprocessing as mp
import datasets.distributed

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        worker_info = None
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}
       
def collate_fn(batch_list):
    batch = {
        "input_ids": torch.stack([torch.Tensor(example["input_ids"]).long() for example in batch_list]),
        "attention_mask": torch.stack([torch.Tensor(example["attention_mask"]).long() for example in batch_list]),
    }
    return batch

def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch
        
def evaluate_model(model, preprocess_batched, pad_idx, rank, world_size, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True)
    val_data = val_data.shuffle(seed=42)
    val_data = datasets.distributed.split_dataset_by_node(val_data, rank=rank, world_size=world_size)
    print(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: batch_fn(val_data_mapped, batch_size)

    target_batchs = 100
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    print(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if total_batches > target_batchs:
            break
        total_batches += world_size

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels, use_cache=False).loss
        total_loss += loss.detach()

    total_loss = total_loss / total_batches
    
    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) 

    return total_loss, evaluated_on_tokens 

RPC_PORT = 29501

    
def benchmark_dp(rank, args, world_size):
    """Benchmark a given model using a single process and multiple devices."""
    init_method_pgroup = "tcp://localhost:{}".format(RPC_PORT)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method=init_method_pgroup
    )
    
    torch.cuda.set_device(rank)
    # Specify the pretrained model name or path
    model_name = args.model_name
    
    # Load the tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
    model_config = AutoConfig.from_pretrained(args.model_config)
    model = LlamaForCausalLM(model_config)
    tokenizer.pad_token = tokenizer.eos_token
    pad_idx = tokenizer.pad_token_id
    
    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = DDP(model)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Instantiate the dataset
    num_samples = args.num_samples  # Number of random samples you want to generate
    max_length = args.max_length  # Maximum length of the sequence

    # Load the "allenai/c4" dataset with streaming=True
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = datasets.distributed.split_dataset_by_node(
        dataset, rank=rank, world_size=world_size,
    )
    dataset = PreprocessedIterableDataset(dataset, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    step = 0
    num_epochs = 3
    update_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:

            step += 1
            
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
    
            outputs = model(**batch, labels=labels, use_cache=False)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % args.eval_every == 0:
                print(f"Performing evaluation at step {step}")
                total_loss, evaluated_on_tokens = evaluate_model(
                    model, preprocess_batched, pad_idx, rank, world_size, device, args.batch_size
                )
                print(total_loss)
    
                print(
                    "Peak allocated bytes on cuda:{}: {:4f}GB".format(
                        dist.get_rank(), torch.cuda.memory_stats(dist.get_rank())["allocated_bytes.all.peak"] / 2**30
                    )
                )
                    
                print(time.time() - update_time)
                
                update_time = time.time()
                if step // args.eval_every == 2:
                    return() 
    
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    print(
        "Peak allocated bytes on {:4f}GB".format(
            torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] / 2**30
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config", type=str, default="configs/llama_60m.json"
    )
    parser.add_argument(
        "--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16
    )
    parser.add_argument(
        "--num_samples", type=int, default=10
    )
    parser.add_argument(
        "--max_length", type=int, default=512
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=100)
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