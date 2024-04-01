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
import torch.nn.functional as F

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
        is_MultiheadAttention = isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention) or isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP) or isinstance(module, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoMLP)
        is_linear = nn.Linear

        if has_child and not is_MultiheadAttention:
            for name, child in module.named_children():
                m = RecursiveVisit(name, child, module)
                if isinstance(m, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoMLP):
                    return m
        else:
            return module

    attention = RecursiveVisit('name', model, model)
    print(attention.c_fc)
    attention.embed_dim = attention.c_fc.in_features
    attention = copy.deepcopy(attention)

    del model
    print(attention)
    
    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention.to(device)
    # attention.eval()
    
    num_epochs = 1
    split = 8
    batch = torch.randn(args.batch_size, args.max_length, attention.embed_dim).cuda()
    inputs = batch.to(device)
    
    with torch.no_grad():
        for epoch in range(num_epochs):
            start_time = time.time()
            output_ref = attention(inputs)
    
            output_parallel = None
    
            for i in range(split):
                slice1 = slice(
                    attention.c_fc.out_features // split * i,
                    attention.c_fc.out_features // split * (i+1),
                )
                outputs =  attention.dropout(F.linear(attention.act(F.linear(inputs, attention.c_fc.weight[slice1],  attention.c_fc.bias[slice1])), attention.c_proj.weight[:, slice1],  attention.c_proj.bias/split))
    
                if output_parallel == None:
                    output_parallel = outputs
                else:
                    output_parallel += outputs
            
            # # output_parallel.mean().backward()
            assert torch.allclose(output_ref, output_parallel, atol=1e-3), f"{torch.max(output_ref-output_parallel), output_ref,output_parallel}"

            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f} seconds")
    
            print(output_parallel)


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