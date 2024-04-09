import argparse
import os
import time

import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torch.optim import AdamW

from utils import load, load_jsonl, load_data
from datasets import load_dataset, load_from_disk

from transformers import TrainingArguments, TextDataset, DataCollatorForLanguageModeling

import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import CustomPolicy
import torch.multiprocessing as mp
import torch.distributed as dist

import transformers
from functools import partial

RPC_PORT = 29501

def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def lambda_fn(module: nn.Module):
    if isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention) or isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2MLP):
        return True
    return False


def hook_fn(param, *unused):
    if param.grad is None:
        return
    print('test', param.grad.shape)
    param.lora_grad = param.grad[0:5]
    param.grad = None

class warpper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
    ):
        super().__init__()
        self.module = module
        
    def forward(self, *args, **kwargs):
        outputs = self.module(*args, **kwargs)
        self._register_post_backward_hooks()
        return outputs

    def _register_post_backward_hooks(self) -> None:
        if not torch.is_grad_enabled():
            return  # don't register grad hooks if grad isn't enabled
        for p in self.module.parameters():
            if p.requires_grad:
                # Register a hook.
                p_tmp = p.expand_as(p)  # Get a grad_fn on p_tmp.
                assert p_tmp.grad_fn is not None
                grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.

                handle = grad_acc.register_hook(partial(hook_fn, p))
                p._post_backward_hook_state = (grad_acc, handle)

# def RecursiveVisit(name, module, upper_module):
    
#     """
#     Recursively replace layers in the module with the custom layer.
    
#     Args:
#     - module (nn.Module): The module (or model) to modify.
#     - custom_layer_class (nn.Module): The custom layer class to replace with.
#     """
        
#     has_child = any(isinstance(child, nn.Module) for child in module.children())
#     is_cusomized = isinstance(module, FullyShardedDataParallel) and not (module is upper_module)

#     if has_child and not is_cusomized:
#         for name, child in module.named_children():
#             RecursiveVisit(name, child, module)
#     else:
#         if is_cusomized:
#             module = warpper(module)
#             setattr(upper_module, name, module)

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

    policy = CustomPolicy(lambda_fn)
    model = FullyShardedDataParallel(model, auto_wrap_policy=policy)
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
    
    # Set up the optimizer
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
    
        for batch in data_loader:
            inputs = batch.to(device)
            outputs = model(input_ids=inputs, labels=inputs, use_cache=False)
            loss = outputs.loss
            loss.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
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