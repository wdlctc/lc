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


from triton import heuristics, jit
from triton import language as tl
from triton import next_power_of_2

import triton



def num_warps(N):
    if N < 2048:
        return 4
    elif N < 8192:
        return 8
    return 16



def num_warps(N):
    if N < 2048:
        return 4
    elif N < 8192:
        return 8
    return 16

def num_warps(N):
    if N < 2048:
        return 4
    elif N < 8192:
        return 8
    return 16


@heuristics({'num_warps': lambda nargs: num_warps(nargs['N'])})
@heuristics({'BLOCK': lambda nargs: next_power_of_2(nargs['N'])})
@jit
def _forward(LOGITS, PROBS, IDX, LOSS, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    idx = tl.load(IDX + row)
    ignore_index = -100
    # pointers to logit and probs
    LOGITS = LOGITS + row * N + cols
    WRIT_PROBS = PROBS + row * N + cols
    READ_PROBS = PROBS + row * N + idx
    # write-back negative log-probs
    logits = tl.load(LOGITS, mask=cols < N, other=-float('inf'))
    logits = logits.to(tl.float32)
    logits = logits - tl.max(logits, 0)
    probs = tl.log(tl.sum(tl.exp(logits), 0)) - logits

    probs_loss = tl.log(tl.sum(tl.exp(logits), 0)) - tl.sum(tl.where(cols == idx, logits, 0.0))
    probs_loss = tl.where(idx == ignore_index, 0.0, probs_loss)
    # tl.store(WRIT_PROBS, probs, mask=cols < N)

    # There is a bug in the compiler, which fails to insert a barrier here.
    # We add it explicitly for now. Will be fixed soon.
    # tl.debug_barrier()
    # write-back loss
    # probs_loss = tl.load(READ_PROBS)
    # probs_loss = tl.where(idx == ignore_index, 0.0, probs_loss)
    tl.store(LOSS + row, probs_loss)

    tl.debug_barrier()
    probs = -probs
    probs = tl.exp(probs.to(tl.float32))
    delta = cols == idx
    din = (probs - delta)
    din = tl.where(idx == ignore_index, 0.0, din)
    tl.store(WRIT_PROBS, din, mask=cols < N)

class _cross_entropy(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, hidden_states, indices, weights):
        logits = torch.matmul(hidden_states, weights.T)
        # make sure we can use triton
        assert (indices.dtype == torch.int64), "Indices are expected to be of type long."
        # make kernel
        device, dtype = logits.device, logits.dtype
        n_cols = logits.shape[-1]
        # run the kernel
        result = torch.empty_like(indices, dtype=dtype, device=device)
        neg_logprobs = torch.empty_like(logits, dtype=dtype, device=device)
        grid = lambda opt: (logits.numel() // n_cols, )
        _forward[grid](logits, neg_logprobs, indices, result, n_cols)
        # save for backward

        grad_input = neg_logprobs @ weights

        if hasattr(weights, 'grad') and weights.grad != None:
            torch.addmm(
                    weights.grad,
                    neg_logprobs.T,
                    hidden_states,
                    out=weights.grad,
                )
        else:
            weights.grad = neg_logprobs.T @ hidden_states
        weights.grad_mul = False
        neg_logprobs = None

        ctx.save_for_backward(grad_input, weights)
        result = result.sum()
        return result

    @classmethod
    def backward(cls, ctx, dneg_logprobs):
        """We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
        so we initialize the gradient as neg_logprobs, so we can just exponentiate
        to get p[k], which is most of what we need...  neg_logprobs will be
        modified in place to become the gradient we want
        """
        # load saved tensors
        neg_logprobs, weights = ctx.saved_tensors
        if weights.grad_mul is False:
            weights.grad *= dneg_logprobs
            weights.grad_mul = True
        neg_logprobs *= dneg_logprobs
        
        return neg_logprobs, None, weights.grad


class FusedCrossEntropyLMhead(nn.Module):
    def __init__(
        self,
        original_weight = None
    ):
        super().__init__()
        if original_weight is None:
            self.LM_head_weight = nn.Parameter(torch.empty(hidden_size, vocab_size))
        else:
            self.LM_head_weight = original_weight
        self.cross_entropy = _cross_entropy.apply

    def forward(self, hidden_states, labels):
        loss = self.cross_entropy(hidden_states, labels, self.LM_head_weight)
        return loss

pretraining_tp = 2

def narrow_processing(hidden_states, labels, lm_head):

    bsz, q_len, hidden_size = hidden_states.size()
    tmp = q_len // pretraining_tp


    hidden_states = hidden_states[..., :-1, :]

    labels = labels[..., 1:].contiguous()
    labels = labels.to(hidden_states.device)
    
    loss = None
    for i in range(pretraining_tp):
        Fused = FusedCrossEntropyLMhead(lm_head.weight)

        shift_hidden_states = hidden_states[..., i * tmp : (i+1)*tmp, :].contiguous()
        shift_hidden_states = shift_hidden_states.view(-1, hidden_size)
        shift_labels = labels[..., i * tmp : (i+1)*tmp ].contiguous()
        shift_labels = shift_labels.view(-1)

        loss_i = Fused(shift_hidden_states, shift_labels)

        if not torch.isnan(loss_i):
            if loss is None:
                loss = loss_i
            else:
                loss = loss + loss_i
        # print(i, loss_i, loss, labels, hidden_states)

    loss = loss / torch.sum(torch.ne(labels, -100))
    return None, loss


from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

def narrow_processing2(hidden_states, labels, lm_head):

    bsz, q_len, hidden_size = hidden_states.size()


    labels = labels[..., 1:].contiguous()
    labels = labels.to(hidden_states.device)
    
    loss = None
    logits = lm_head(hidden_states)
    logits = logits[..., :-1, :].contiguous()
    logits = logits.view(-1, 128256)
    labels = labels.view(-1)
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(logits, labels)
    return None, loss



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

    lm_head = model.lm_head

    del model

    print(lm_head)
    
    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_head.to(device)
    
    num_epochs = 3
    batch = torch.randn(args.batch_size, args.max_length, 4096, dtype=torch.float16).cuda()
    label = 4 + torch.ones((args.batch_size, args.max_length), dtype=torch.int64, device=device)
    inputs = batch.to(device)
    
    dtype, device = inputs.dtype, inputs.device
    min_dtype = torch.finfo(dtype).min

    # mlp.gate_proj.weight.requires_grad = False
    # mlp.up_proj.weight.requires_grad = False
    # mlp.down_proj.weight.requires_grad = False
    # inputs.requires_grad = True
    
    torch.cuda.synchronize()
    for epoch in range(num_epochs):
        init_random_seed(epoch)
        start_time = time.time()
        for i in range(10):
            _, loss = narrow_processing(batch, label, lm_head)
            loss.backward()
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
        "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B"
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
        "--max_length", type=int, default=1024
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