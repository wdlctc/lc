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

def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[1] = dim_size[1] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(
        output, input_.contiguous(), group=group
    )

    return output

def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert (
        dim_size[1] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[1] = dim_size[1] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(
        output, input_.contiguous(), group=group
    )
    return output
    
def _split_along_last_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    rank = torch.distributed.get_rank(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    input_list = split_tensor(input_, world_size)

    output = input_list[rank].contiguous()

    return output
    
def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[-1] = dim_size[-1] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(
        output, input_.contiguous(), group=group
    )

    return output
    
class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)

class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad
        return _reduce_scatter_along_first_dim(grad_output), None
        
class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)

class SequenceWarpper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        group: Optional[Any] = None,
    ):
        super().__init__()
        self.module = module
        self.group = group
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:

        inputs = _GatherFromSequenceParallelRegion.apply(args[0])
        outputs = self.module(inputs, **kwargs)
        if isinstance(outputs, tuple):
            outputs = list(outputs)
            outputs[0] = _ReduceScatterToSequenceParallelRegion.apply(outputs[0])
            outputs = tuple(outputs)
        else:
            outputs = _ReduceScatterToSequenceParallelRegion.apply(outputs)
            
        return outputs
        
def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()
    
class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input_, ctx.world_size, process_group, scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )

def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)
    
class EmbedderWarpper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        group: Optional[Any] = None,
    ):
        super().__init__()
        self.module = module
        self.group = group
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        print('EmbedderWarpper', outputs.shape)
        outputs = all_to_all(outputs, self.group, scatter_dim=1, gather_dim=-1)
        print('EmbedderWarpper', outputs.shape)
        return outputs

class AllgatherWarpper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        group: Optional[Any] = None,
    ):
        super().__init__()
        self.module = module
        self.group = group
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        outputs = _GatherFromModelParallelRegion.apply(outputs)
        return outputs


class GPT2Attention(nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.config = ref.config
        max_positions = ref.config.max_position_embeddings
        self.bias = ref.bias
        self.masked_bias = ref.masked_bias

        self.embed_dim = ref.embed_dim
        self.num_heads = ref.num_heads
        self.head_dim = ref.head_dim
        self.split_size = ref.split_size
        self.scale_attn_weights = ref.scale_attn_weights
        self.is_cross_attention = ref.is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = ref.scale_attn_by_inverse_layer_idx
        self.layer_idx = ref.layer_idx
        self.reorder_and_upcast_attn = ref.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = ref.c_attn
            self.q_attn = ref.q_attn
        else:
            self.c_attn = ref.c_attn
        self.c_proj = ref.c_proj 

        self.attn_dropout = ref.attn_dropout
        self.resid_dropout = ref.resid_dropout

        self.pruned_heads = ref.pruned_heads
    
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class SequenceParallel(nn.Module):
    def __init__(
        self,
        module,
        group: Optional[Any] = None):
        super().__init__()
        
        self.module = module
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        
        self.FlyweightModule_list = []
        self.RecursiveVisit('module', self.module, self.module)
        self.eval= False
        
    def RecursiveVisit(self, name, module, upper_module):
        """
        Recursively replace layers in the module with the custom layer.
        
        Args:
        - module (nn.Module): The module (or model) to modify.
        - custom_layer_class (nn.Module): The custom layer class to replace with.
        """
            
        has_parameters = any(isinstance(param, nn.Parameter) for param in module.parameters())
        has_child = any(isinstance(child, nn.Module) for child in module.children())
        is_MultiheadAttention = isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention) or isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP) or isinstance(module, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoMLP) or isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention) or isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2MLP)
        is_linear = nn.Linear

        if has_child and not is_MultiheadAttention:
            for name, child in module.named_children():
                self.RecursiveVisit(name, child, module)
        else:
            if isinstance(module, nn.Embedding):
                if module.embedding_dim % self.world_size == 0:
                    module.weight = nn.Parameter(split_tensor(module.weight, self.world_size, dim=1)[self.rank])
                    module = FlyweightWarpper(module, self.group)
                    setattr(upper_module, name, module)
                    self.FlyweightModule_list.append(module)
                    
            elif isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention):
                module = GPT2Attention(module)
                setattr(upper_module, name, module)
                # if module.c_attn is not None:
                #     module.c_attn.weight = nn.Parameter(split_tensor(module.c_attn.weight, self.world_size, dim=1)[self.rank])
                #     if module.c_attn.bias is not None:
                #         module.c_attn.bias = nn.Parameter(split_tensor(module.c_attn.bias, self.world_size, dim=0)[self.rank])
                #         module.c_attn.nf = module.c_attn.nf // self.world_size
                # if hasattr(module, 'q_attn'):
                #     module.q_attn.weight = nn.Parameter(split_tensor(module.q_attn.weight, self.world_size, dim=1)[self.rank])
                #     if module.q_attn.bias is not None:
                #         module.q_attn.bias = nn.Parameter(split_tensor(module.q_attn.bias, self.world_size, dim=0)[self.rank])
                #         module.q_attn.nf = module.q_attn.nf // self.world_size
                # module.num_heads = module.num_heads // self.world_size
                # module.split_size = module.split_size // self.world_size

                # module.c_proj.weight = nn.Parameter(split_tensor(module.c_proj.weight, self.world_size, dim=0)[self.rank])
                # if module.c_proj.bias is not None:
                #     module.c_proj.bias.data.div_(self.world_size)
                # module = FlyweightWarpper(module, self.group, cat_output=False)

                # setattr(upper_module, name, module)

                # self.FlyweightModule_list.append(module)
            elif isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2MLP):
                module.c_fc.weight = nn.Parameter(split_tensor(module.c_fc.weight, self.world_size, dim=1)[self.rank])
                if module.c_fc.bias is not None:
                    module.c_fc.bias = nn.Parameter(split_tensor(module.c_fc.bias, self.world_size, dim=0)[self.rank])
                module.c_fc.nf = module.c_fc.nf // self.world_size
                    
                module.c_proj.weight = nn.Parameter(split_tensor(module.c_proj.weight, self.world_size, dim=0)[self.rank])
                module = FlyweightWarpper(module, self.group, cat_output=False)
                setattr(upper_module, name, module)
                self.FlyweightModule_list.append(module)
                
            elif isinstance(module, nn.Linear):
                if module.out_features % self.world_size == 0:
                    module.weight = nn.Parameter(split_tensor(module.weight, self.world_size, dim=0)[self.rank])
                    if module.bias is not None:
                        module.bias = nn.Parameter(split_tensor(module.bias, self.world_size, dim=0)[self.rank])
                    module = FlyweightWarpper(module, self.group, inplace=self.inplace)
                elif module.in_features % self.world_size == 0:
                    module.weight = nn.Parameter(split_tensor(module.weight, self.world_size, dim=-1)[self.rank])
                    if module.bias is not None:
                        module.bias.data.div_(self.world_size)
                    module = FlyweightWarpper(module, self.group, row_partition=True, input_partition_dim=-1)
                else:
                    raise ValueError("The input or output features of the linear layer must be divisible by the world size.")
                
                setattr(upper_module, name, module)

                self.FlyweightModule_list.append(module)
            # if isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention):
            #     if module.c_attn is not None:
            #         module.c_attn.weight = nn.Parameter(split_tensor(module.c_attn.weight, self.world_size, dim=1)[self.rank])
            #         if module.c_attn.bias is not None:
            #             module.c_attn.bias = nn.Parameter(split_tensor(module.c_attn.bias, self.world_size, dim=0)[self.rank])
            #             module.c_attn.nf = module.c_attn.nf // self.world_size
            #     if hasattr(module, 'q_attn'):
            #         module.q_attn.weight = nn.Parameter(split_tensor(module.q_attn.weight, self.world_size, dim=1)[self.rank])
            #         if module.q_attn.bias is not None:
            #             module.q_attn.bias = nn.Parameter(split_tensor(module.q_attn.bias, self.world_size, dim=0)[self.rank])
            #             module.q_attn.nf = module.q_attn.nf // self.world_size
            #     module.num_heads = module.num_heads // self.world_size
            #     module.split_size = module.split_size // self.world_size

            #     module.c_proj.weight = nn.Parameter(split_tensor(module.c_proj.weight, self.world_size, dim=0)[self.rank])
            #     module = SequenceWarpper(module, self.group)
            #     setattr(upper_module, name, module)
            # elif isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2MLP):
            #     module.c_fc.weight = nn.Parameter(split_tensor(module.c_fc.weight, self.world_size, dim=1)[self.rank])
            #     if module.c_fc.bias is not None:
            #         module.c_fc.bias = nn.Parameter(split_tensor(module.c_fc.bias, self.world_size, dim=0)[self.rank])
            #     module.c_fc.nf = module.c_fc.nf // self.world_size
                    
            #     module.c_proj.weight = nn.Parameter(split_tensor(module.c_proj.weight, self.world_size, dim=0)[self.rank])
            #     module = SequenceWarpper(module, self.group)
            #     setattr(upper_module, name, module)
            
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        if not self.eval:
            self._register_post_backward_hooks()
        return outputs

    def _register_post_backward_hooks(self) -> None:
        if not torch.is_grad_enabled():
            return  # don't register grad hooks if grad isn't enabled
        for module in self.FlyweightModule_list:
            for i, sub_module in enumerate(module.module_list):
                if i == self.rank:
                    continue
                sub_module.count = 0
                for p in sub_module.parameters():
                    if p.requires_grad:
                        # Register a hook.
                        p_tmp = p.expand_as(p)  # Get a grad_fn on p_tmp.
                        assert p_tmp.grad_fn is not None
                        grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.
                        sub_module.count += 1

                        if  not hasattr(p, '_my_handle'):
                            handle = grad_acc.register_hook(partial(hook_fn, sub_module, module))
                            p._my_handle = handle
    
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
    
    model = SequenceParallel(model)
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
    print(model)
    
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
            outputs = model(input_ids=inputs, labels=inputs)
            loss = outputs.loss
            loss.backward()
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
        "--model_name", type=str, default="openai-community/gpt2-medium"
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