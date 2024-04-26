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

from transformers.cache_utils import Cache, DynamicCache, StaticCache

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
    model.eval()

    
    def RecursiveVisit(name, module, upper_module):
        """
        Recursively replace layers in the module with the custom layer.
        
        Args:
        - module (nn.Module): The module (or model) to modify.
        - custom_layer_class (nn.Module): The custom layer class to replace with.
        """
            
        has_parameters = any(isinstance(param, nn.Parameter) for param in module.parameters())
        has_child = any(isinstance(child, nn.Module) for child in module.children())
        is_MultiheadAttention = isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention)

        if has_child and not is_MultiheadAttention:
            for name, child in module.named_children():
                m = RecursiveVisit(name, child, module)
                if isinstance(m, transformers.models.llama.modeling_llama.LlamaAttention):
                    return m
        else:
            return module

    attention = RecursiveVisit('name', model, model)
    attention.training = True
    
    del model
    
    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention.to(device)
    attention2 = copy.deepcopy(attention)
    
    position_ids = torch.arange(
        0, args.max_length, device=device
    ).unsqueeze(0)
    
    # temp_mask = torch.ones(args.max_length // 2, args.max_length // 2, dtype=torch.bool).tril(diagonal=0).cuda().unsqueeze(0).unsqueeze(0)
    # full_mask = torch.ones(args.max_length // 2, args.max_length // 2, dtype=torch.bool).cuda().unsqueeze(0).unsqueeze(0)

    temp_mask = None


    def get_mini_sequence_list(max_length, mini_sequence):
        
        position_ids_list = []
        sub_length = args.max_length // mini_sequence
        for i in range(mini_sequence):
            position_ids_list.append(torch.arange(
                sub_length * i, sub_length * (i+1), device=device
            ).unsqueeze(0))

        
        mask_list = []
        
        temp_mask = torch.ones(sub_length, sub_length, dtype=torch.bool).tril(diagonal=0).cuda().unsqueeze(0).unsqueeze(0).expand(args.batch_size, 1, sub_length, sub_length)
        full_mask = torch.ones(sub_length, sub_length, dtype=torch.bool).cuda().unsqueeze(0).unsqueeze(0).expand(args.batch_size, 1, sub_length, sub_length)
        all_mask = temp_mask

    
        for i in range(mini_sequence):
            mask_list.append(all_mask)
            all_mask = torch.cat((full_mask, all_mask), dim=-1)

        return position_ids_list, mask_list

    
    mini = 4
    position_ids_list, mask_list = get_mini_sequence_list(args.max_length, mini)
    
    num_epochs = 3
    batch = torch.randn(args.batch_size, args.max_length, attention.hidden_size, dtype=torch.float16).cuda()
    seqinputs = batch.to(device).clone().detach()
    torch.cuda.synchronize()
    for epoch in range(num_epochs):
        init_random_seed(epoch)
        start_time = time.time()
        # seqinputs = inputs.clone().detach()

        ref =  attention2(hidden_states=batch, position_ids=position_ids)
        ref[0].backward(ref[0])

        past = DynamicCache()

        output_list = []

        for i in range(mini):
            print(i)
            outputs = attention(hidden_states=seqinputs[:, i*(args.max_length//mini): (i+1)*(args.max_length//mini), :], position_ids=position_ids[:, i*(args.max_length//mini): (i+1)*(args.max_length//mini)], use_cache=True, past_key_value=past, attention_mask=mask_list[i])

            assert torch.allclose(outputs[0], ref[0][:, i*(args.max_length//mini): (i+1)*(args.max_length//mini), :], atol=1e-3), f"{outputs[0], ref[0][:, i*(args.max_length//mini): (i+1)*(args.max_length//mini), :]}"
            
            # outputs[0].backward(outputs[0])
            output_list.append(outputs[0])

        for i in range(mini-1, -1, -1):
            print(i)
            output_list[i].backward(output_list[i])
        #     print(i)
        #     if i == mini-1:
        #         outputs[0].backward(outputs[0])
        #     else:
        #         for j in range(len(past)):
        #             print(past.key_cache[j].shape)
        #             past.key_cache[j] = past.key_cache[j][:,:,(i-1)*(args.max_length//mini): (i)*(args.max_length//mini),:].detach()
        #             past.value_cache[j] = past.value_cache[j][:,:,(i-1)*(args.max_length//mini): (i)*(args.max_length//mini),:].detach()
        #         outputs = attention(hidden_states=seqinputs[:, i*(args.max_length//mini): (i+1)*(args.max_length//mini), :], position_ids=position_ids[:, i*(args.max_length//mini): (i+1)*(args.max_length//mini)], use_cache=True, past_key_value=past, attention_mask=temp_mask)
        #         outputs[0].backward(outputs[0])

        for p1, p2 in zip(attention.named_parameters(), attention2.named_parameters()):
            assert torch.allclose(p1[1].grad, p2[1].grad, rtol=1e-3, atol=1e-4), f"\n{p1[0]}\nvs\n{p2[0]}:\n{p1[1].grad}\nvs\n{p2[1].grad}"
            p1[1].grad = None
            p2[1].grad = None

        torch.cuda.synchronize()
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
        "--max_length", type=int, default=16
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