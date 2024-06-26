import argparse
import os
import time
import torch.nn as nn

import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

from torch.optim import AdamW

from utils import load, load_jsonl, load_data
from datasets import load_dataset, load_from_disk

from transformers import TrainingArguments, TextDataset, DataCollatorForLanguageModeling 
import transformers

import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist

from rtp.rotated_tensor_parallel import RotatedTensorParallel

import copy

RPC_PORT = 29501

def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def _gather(input_: torch.Tensor, dim) -> torch.Tensor:

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size() == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output

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

    
    def RecursiveVisit(name, module, upper_module):
        """
        Recursively replace layers in the module with the custom layer.
        
        Args:
        - module (nn.Module): The module (or model) to modify.
        - custom_layer_class (nn.Module): The custom layer class to replace with.
        """
            
        has_parameters = any(isinstance(param, nn.Parameter) for param in module.parameters())
        has_child = any(isinstance(child, nn.Module) for child in module.children())
        is_MultiheadAttention = isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention) or isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP) 

        if has_child and not is_MultiheadAttention:
            for name, child in module.named_children():
                m = RecursiveVisit(name, child, module)
                if isinstance(m, nn.Linear):
                    return m
        else:
            return module

    attention = RecursiveVisit('name', model, model)
    attention.embed_dim = attention.in_features
    attention = copy.deepcopy(attention)
    ddp_attention = copy.deepcopy(attention)
    fsdp_attention = copy.deepcopy(attention)
    rtp_attention = copy.deepcopy(attention)

    del model
    print(attention)
    
    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention.to(device)
    ddp_attention = DDP(ddp_attention.to(device))
    fsdp_attention = FullyShardedDataParallel(fsdp_attention.to(device))
    rtp_attention = RotatedTensorParallel(rtp_attention.to(device), inplace=False)
    rtp_attention.eval()
    
    num_epochs = 3
    for epoch in range(num_epochs):
        start_time = time.time()
        batch = torch.randn(args.batch_size, args.max_length, attention.embed_dim, dtype=torch.float16).cuda()
        inputs = batch.to(device)
        outputs = attention(inputs)
        DDP_outputs = ddp_attention(inputs)
        fsdp_outputs = fsdp_attention(inputs)
        rtp_outputs = rtp_attention(inputs)

        assert torch.allclose(outputs[0], rtp_outputs[0], atol=1e-5), f"{torch.max(outputs[0]-rtp_outputs[0])}"

        outputs.mean().backward()
        rtp_outputs.mean().backward()
        Weight_loss = criterion(Weight_linear_output, cur_label) / world_size
        Weight_loss.backward()
        
        # all reduce gradient for sp
        for p in rtp_attention.parameters():
            p.grad.data = _gather(p.grad, dim=0)
    
        # check grad
        for p1, p2 in zip(rtp_attention.parameters(), attention.parameters()):
            assert torch.allclose(p1.grad, p2.grad, atol=1e-3), f"{p1.grad}\nvs\n{p2.grad}"

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f} seconds")


    print(
        "Peak allocated bytes on cuda:{}: {:4f}GB".format(
            dist.get_rank(), torch.cuda.memory_stats(dist.get_rank())["allocated_bytes.all.peak"] / 2**30
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="EleutherAI/gpt-neo-125m"
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