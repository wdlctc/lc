import argparse
import os
import time

import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torch.optim import AdamW

from utils import load, load_jsonl, load_data
from datasets import load_dataset, load_from_disk

from transformers import TrainingArguments, TextDataset, DataCollatorForLanguageModeling

import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist

import transformers

from functools import partial
from rtp.rotated_tensor_parallel import FlyweightWarpper, hook_fn

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

import copy

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
        is_MultiheadAttention = isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention) or isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention)

        if has_child and not is_MultiheadAttention:
            for name, child in module.named_children():
                m = RecursiveVisit(name, child, module)
                if isinstance(m, transformers.models.gpt2.modeling_gpt2.GPT2Attention) or isinstance(m, transformers.models.llama.modeling_llama.LlamaAttention):
                    return m
        else:
            return module
            
    attention = RecursiveVisit('name', model, model)
    attention.training = True

    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention.to(device)

    from sp import OriSequenceParallel, TpSequenceParallel, UlyssesParallel, RtpParallel

    # orisqattention = OriSequenceParallel(copy.deepcopy(attention))
    # tpsqattention = TpSequenceParallel(copy.deepcopy(attention))
    # ulyssattention = DDP(UlyssesParallel(copy.deepcopy(attention)))
    rtpattention = RtpParallel(copy.deepcopy(attention))
    # model = SequenceParallel(model)
    # optimizer = AdamW(model.parameters(), lr=5e-5)

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

    position_ids = torch.arange(
        0, max_length, device=device
    ).unsqueeze(0)
    
    cache_position = torch.arange(
        0, max_length, device=device
    )
    # Set up the optimizer
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        init_random_seed(epoch)
        start_time = time.time()
        batch = torch.randn(args.batch_size, args.max_length, attention.hidden_size, dtype=torch.float16).cuda()
        inputs = batch.to(device)
        inputs.data.div_(5)
        seq_inputs =  split_tensor(inputs, world_size, dim=1)[rank].clone().detach()

        
        inputs.requires_grad = True
        inputs.retain_grad()
        seq_inputs.requires_grad = True
        seq_inputs.retain_grad()
        
        outputs = attention(hidden_states=inputs, position_ids=position_ids)

        output_list = split_tensor(outputs[0], world_size, dim=1)
        # ref = orisqattention(seq_inputs, position_ids=position_ids)[0]
        # ref = tpsqattention(seq_inputs, position_ids=position_ids)[0]
        # ref = ulyssattention(seq_inputs, position_ids=position_ids)[0]
        # ref = rtpattention(seq_inputs, position_ids=position_ids)[0]
        ref = rtpattention(seq_inputs, position_ids=position_ids)[0]

        # print(output_list[rank], ref)
        assert torch.allclose(output_list[rank], ref, atol=1e-3), f"{torch.max((output_list[rank] - ref))}"

        inputs.retain_grad()
        seq_inputs.retain_grad()
        
        outputs[0].mean().backward()
        ref[0].mean().backward()


        # ## oritp
        # for name, p in orisqattention.named_parameters():
        #     if p.grad is not None:
        #         dist.all_reduce(p.grad, group=dist.group.WORLD)
        #         p.grad.div_(world_size * 2)
        #     else:
        #         print(f"grad of {name} is None")
        
        # for p1, p2 in zip(orisqattention.named_parameters(), attention.named_parameters()):
        #     # print(p1[0], p1[1].grad.shape, p2[1].grad.shape, torch.allclose(p1[1].grad, p2[1].grad, rtol=1e-3, atol=1e-4))
        #     assert torch.allclose(p1[1].grad, p2[1].grad, rtol=1e-3, atol=1e-4), f"\n{p1[0]}\nvs\n{p2[0]}:\n{p1[1].grad}\nvs\n{p2[1].grad}"
        #     # print(p1[1].grad, p2[1].grad)
        #     p1[1].grad = None
        #     p2[1].grad = None

        # # seqtp
        # for p1, p2 in zip(tpsqattention.named_parameters(), attention.named_parameters()):

        #     tp_dim = None
        #     for i, dim in enumerate(p1[1].grad.shape):
        #         if dim != p2[1].grad.shape[i]:
        #             tp_dim = i
        #             break

        #     if 'c_attn' in p1[0]:
        #         ref_list = split_tensor(p2[1].grad, world_size * 3, dim=tp_dim)
        #         ref = torch.cat([ref_list[rank + i*world_size] for i in range(3)], dim = tp_dim).mul_(2)
        #     elif tp_dim != None:
        #         ref = split_tensor(p2[1].grad, world_size, dim=tp_dim)[rank].mul_(2)
        #     else:
        #         ref = p2[1].grad.mul_(2)

            
        #     print(p1[0], p1[1].grad.shape,  p2[1].grad.shape)
        #     assert torch.allclose(p1[1].grad, ref, rtol=1e-3, atol=1e-4), f"\n{p1[0]}\nvs\n{p2[0]}:\n{p1[1].grad}\nvs\n{ref}"
        #     # print(p1[1].grad, ref)
        #     p1[1].grad = None
        #     p2[1].grad = None
            
        # # ulysses
        # for name, p in ulyssattention.named_parameters():
        #     if p.grad is not None:
        #         dist.all_reduce(p.grad, group=dist.group.WORLD)
        #         p.grad.div_(world_size)
        #     else:
        #         print(f"grad of {name} is None")
        
        # for p1, p2 in zip(ulyssattention.named_parameters(), attention.named_parameters()):
        #     # print(p1[0], p1[1].grad.shape, p2[1].grad.shape, torch.allclose(p1[1].grad, p2[1].grad, rtol=1e-3, atol=1e-4))
        #     assert torch.allclose(p1[1].grad, p2[1].grad, rtol=1e-3, atol=1e-4), f"\n{p1[0]}\nvs\n{p2[0]}:\n{p1[1].grad}\nvs\n{p2[1].grad}"
        #     # print(p1[1].grad, p2[1].grad)
        #     p1[1].grad = None
        #     p2[1].grad = None
            
        # rtp
        # for p1, p2 in zip(rtpattention.named_parameters(), attention.named_parameters()):
        #     tp_dim = None
        #     for i, dim in enumerate(p1[1].grad.shape):
        #         if dim != p2[1].grad.shape[i]:
        #             tp_dim = i
        #             break

        #     if tp_dim != None:
        #         ref = split_tensor(p2[1].grad, world_size, dim=tp_dim)[rank].mul_(2)
        #     else:
        #         ref = p2[1].grad
            
        #     # assert torch.allclose(p1[1].grad, ref, rtol=1e-3, atol=1e-4), f"\n{p1[0]}\nvs\n{p2[0]}:\n{p1[1].grad}\nvs\n{ref}"
        #     if not torch.allclose(p1[1].grad, ref, rtol=1e-3, atol=1e-4):
        #         print(f"\n{p1[0]}\nvs\n{p2[0]}:\n{p1[1].grad}\nvs\n{ref}")
        #     # print(p1[1].grad, ref)
        #     p1[1].grad = None
        #     p2[1].grad = None

        # inputs_grad = split_tensor(inputs.grad, world_size, dim=1)[rank].mul_(2)
        # assert torch.allclose(inputs_grad, seq_inputs.grad, atol=1e-5), f"{inputs_grad}\nvs\n{seq_inputs.grad}"
        
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
        "--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
        "--max_length", type=int, default=4
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