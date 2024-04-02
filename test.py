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

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

RPC_PORT = 29501

def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def split_tensor(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False, dim: int = -1
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    dim_size = divide(tensor.size()[dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, dim_size, dim=dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list
    
def split_number(num, parts):
    base = num // parts
    remainder = num % parts
    result = [base] * parts
    
    for i in range(remainder):
        result[i] += 1
    
    return result



class FlyweightWarpper(nn.Module):
    def __init__(
        self,
        sub_models
    ):
        super().__init__()
        self.sub_models = sub_models
        self.module = sub_models[0]

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        output_parallel = None
        for sub_model in self.sub_models:
            outputs = sub_model(*args, **kwargs)

            if output_parallel is None:
                output_parallel = outputs
            else:
                if isinstance(output_parallel, tuple):
                    output_parallel = tuple( tensor1 + tensor2 if tensor1 is not None and tensor2 is not None else None 
                                             for tensor1, tensor2 in zip(output_parallel, outputs))
                else:
                    output_parallel = output_parallel + outputs

        return output_parallel

class OutputWarpper(nn.Module):
    def __init__(
        self,
        module
    ):
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        input = args[0][:,-1,:]
        output_parallel = self.module(input)

        return output_parallel

class gpt2warpper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        split=2
    ):
        super().__init__()
        self.module = module
        self.split = split
        self.FlyweightModule_list = []
        self.RecursiveVisit('module', self.module, self)

    def RecursiveVisit(self, name, module, upper_module):
        
        """
        Recursively replace layers in the module with the custom layer.
        
        Args:
        - module (nn.Module): The module (or model) to modify.
        - custom_layer_class (nn.Module): The custom layer class to replace with.
        """
            
        has_parameters = any(isinstance(param, nn.Parameter) for param in module.parameters())
        has_child = any(isinstance(child, nn.Module) for child in module.children())
        is_cusomized = isinstance(module, nn.MultiheadAttention) or isinstance(module, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention)  or isinstance(module, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoMLP) or isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention)  or  isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP)  

        if has_child and not is_cusomized:
            for name, child in module.named_children():
                self.RecursiveVisit(name, child, module)
        else:
            if has_parameters:
                if isinstance(module, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoMLP):
                    sub_modules = []
                    for i in range(self.split):
                        sub_module = copy.deepcopy(module)

                        sub_module.c_fc.weight = nn.Parameter(split_tensor(module.c_fc.weight, self.split, dim=0)[i])
                        if module.c_fc.bias is not None:
                            sub_module.c_fc.bias = nn.Parameter(split_tensor(module.c_fc.bias, self.split, dim=0)[i])
                        
                        sub_module.c_proj.weight = nn.Parameter(split_tensor(module.c_proj.weight, self.split, dim=1)[i])
                        if sub_module.c_proj.bias is not None:
                            sub_module.c_proj.bias.data.div_(self.split)
                        
                        sub_modules.append(sub_module)

                    module = FlyweightWarpper(sub_modules)
                    setattr(upper_module, name, module)
                    self.FlyweightModule_list.append(module)
                    
                elif isinstance(module, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention):
                    sub_modules = []
                    for i in range(self.split):
                        sub_module = copy.deepcopy(module)
                        
                        sub_module.q_proj.weight = nn.Parameter(split_tensor(module.q_proj.weight, self.split, dim=0)[i])
                            
                        sub_module.k_proj.weight = nn.Parameter(split_tensor(module.k_proj.weight, self.split, dim=0)[i])
                        
                        sub_module.v_proj.weight = nn.Parameter(split_tensor(module.v_proj.weight, self.split, dim=0)[i])
        
                        sub_module.out_proj.weight = nn.Parameter(split_tensor(module.out_proj.weight, self.split, dim=1)[i])
                        if sub_module.out_proj.bias is not None:
                            sub_module.out_proj.bias.data.div_(self.split)
                            
                        sub_module.num_heads = sub_module.num_heads // self.split
                        sub_modules.append(sub_module)

                    module = FlyweightWarpper(sub_modules)
                    setattr(upper_module, name, module)
                    self.FlyweightModule_list.append(module)
                elif isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention):
                    sub_modules = []
                    for i in range(self.split):
                        sub_module = copy.deepcopy(module)
                        sub_module.q_proj.weight = nn.Parameter(split_tensor(module.q_proj.weight, self.split, dim=0)[i])
                        if sub_module.q_proj.bias is not None:
                            sub_module.q_proj.bias = nn.Parameter(split_tensor(module.q_proj.bias, self.split, dim=0)[i])
                        
                        sub_module.k_proj.weight = nn.Parameter(split_tensor(module.k_proj.weight, self.split, dim=0)[i])
                        if sub_module.k_proj.bias is not None:
                            sub_module.k_proj.bias = nn.Parameter(split_tensor(module.k_proj.bias, self.split, dim=0)[i])
                            
                        sub_module.v_proj.weight = nn.Parameter(split_tensor(module.v_proj.weight, self.split, dim=0)[i])
                        if sub_module.v_proj.bias is not None:
                            sub_module.v_proj.bias = nn.Parameter(split_tensor(module.v_proj.bias, self.split, dim=0)[i])
                            
                        sub_module.o_proj.weight = nn.Parameter(split_tensor(module.o_proj.weight, self.split, dim=1)[i])
                        
                        sub_module.num_heads = module.num_heads // self.split
                        sub_module.num_key_value_heads = module.num_key_value_heads // self.split
                        sub_module.hidden_size = module.hidden_size // self.split
                        sub_modules.append(sub_module)

                    module = FlyweightWarpper(sub_modules)
                    setattr(upper_module, name, module)
                    self.FlyweightModule_list.append(module)
                elif isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP):
                    sub_modules = []
                    for i in range(self.split):
                        sub_module = copy.deepcopy(module)
                        sub_module.up_proj.weight = nn.Parameter(split_tensor(module.up_proj.weight, self.split, dim=0)[i])
                            
                        sub_module.gate_proj.weight = nn.Parameter(split_tensor(module.gate_proj.weight, self.split, dim=0)[i])
                        
                        sub_module.down_proj.weight = nn.Parameter(split_tensor(module.down_proj.weight, self.split, dim=1)[i])
                        
                        sub_modules.append(sub_module)
    
                    module = FlyweightWarpper(sub_modules)
                    setattr(upper_module, name, module)
                    self.FlyweightModule_list.append(module)
                elif isinstance(module, nn.Linear):
                    module = OutputWarpper(module)
                    setattr(upper_module, name, module)
                    self.FlyweightModule_list.append(module)
    
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        return outputs
    
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
    model.eval()

    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
    model.to(device)

    # ref_model = copy.deepcopy(model)

    # model = copy.deepcopy(model)

    # model = gpt2warpper(model, split=4)
    print(model)
    
    num_epochs = 5
    init_random_seed(0)
    batch = torch.tensor(np.random.randint(low=0, high=len(tokenizer), size=(args.batch_size, args.max_length,)))
    inputs = batch.to(device)
    with torch.no_grad():
        for epoch in range(num_epochs):
            start_time = time.time()
            output_parallel = model(input_ids=inputs)
            # output_ref = ref_model(input_ids=inputs, past_key_values=None, use_cache=False)
            
            # assert torch.allclose(output_ref.logits, output_parallel.logits, atol=1e-3), f"{torch.max(output_ref.logits-output_parallel.logits), output_ref.logits,output_parallel.logits}"
            output_parallel = None
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
        "--max_length", type=int, default=64
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