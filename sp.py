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

def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    rank = torch.distributed.get_rank(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    input_list = split_tensor(input_, world_size, dim=1)

    output = input_list[rank].contiguous()

    return output
    
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
        return _split_along_first_dim(grad_output), None
        
class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)
    
class OriSequenceWarpper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        group: Optional[Any] = None,
    ):
        super().__init__()
        self.module = module
        self.group = group
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:

        if args:
            inputs = _GatherFromSequenceParallelRegion.apply(args[0])
            outputs = self.module(inputs, **kwargs)
        else:
            kwargs['hidden_states'] = _GatherFromSequenceParallelRegion.apply(kwargs['hidden_states'])
            outputs = self.module(*args, **kwargs)
            
        outputs = list(outputs)
        outputs[0] = _ScatterToSequenceParallelRegion.apply(outputs[0])
        outputs = tuple(outputs)
        return outputs
        
class OriSequenceParallel(nn.Module):
    def __init__(
        self,
        module,
        group: Optional[Any] = None):
        super().__init__()
        
        self.module = module
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        
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
        is_MultiheadAttention = isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention) or isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP) or isinstance(module, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoMLP) or isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention)
        is_linear = nn.Linear

        if has_child and not is_MultiheadAttention:
            for name, child in module.named_children():
                self.RecursiveVisit(name, child, module)
        else:
            if isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention) or isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention):
                module = OriSequenceWarpper(module, self.group)
                setattr(upper_module, name, module)
            
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        return outputs

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
    
class _Gather(torch.autograd.Function):
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
        
class _ReduceScatter(torch.autograd.Function):
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

        if args:
            inputs = _Gather.apply(args[0])
            outputs = self.module(inputs, **kwargs)
        else:
            kwargs['hidden_states'] = _Gather.apply(kwargs['hidden_states'])
            outputs = self.module(*args, **kwargs)
            
        if isinstance(outputs, tuple):
            outputs = list(outputs)
            outputs[0] = _ReduceScatter.apply(outputs[0])
            outputs = tuple(outputs)
        else:
            outputs = _ReduceScatter.apply(outputs)
            
        return outputs
        

class TpSequenceParallel(nn.Module):
    def __init__(
        self,
        module,
        group: Optional[Any] = None):
        super().__init__()
        
        self.module = module
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        
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
        is_MultiheadAttention = isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention) or isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP) or isinstance(module, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoMLP) or isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention) or isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2MLP)
        is_linear = nn.Linear

        if has_child and not is_MultiheadAttention:
            for name, child in module.named_children():
                self.RecursiveVisit(name, child, module)
        else:
            if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
                module = DDP(module)
                setattr(upper_module, name, module)
            elif isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention):
                if module.c_attn is not None:
                    # print(module.c_attn.weight.shape)
                    weight_list = split_tensor(module.c_attn.weight, self.world_size * 3, dim=1)
                    my_weight_list = [weight_list[self.rank + i*self.world_size] for i in range(3)]
                    module.c_attn.weight = nn.Parameter(torch.cat(my_weight_list, dim=1).contiguous())
                    # print(module.c_attn.weight.shape)
                    
                    if module.c_attn.bias is not None:
                        bias_list = split_tensor(module.c_attn.bias, self.world_size * 3, dim=0)
                        my_bias_list = [bias_list[self.rank + i*self.world_size] for i in range(3)]
                        module.c_attn.bias = nn.Parameter(torch.cat(my_bias_list, dim=0).contiguous())
                        module.c_attn.nf = module.c_attn.nf // self.world_size
                if hasattr(module, 'q_attn'):
                    module.q_attn.weight = nn.Parameter(split_tensor(module.q_attn.weight, self.world_size, dim=1)[self.rank])
                    if module.q_attn.bias is not None:
                        module.q_attn.bias = nn.Parameter(split_tensor(module.q_attn.bias, self.world_size, dim=0)[self.rank])
                        module.q_attn.nf = module.q_attn.nf // self.world_size
                module.num_heads = module.num_heads // self.world_size
                module.split_size = module.split_size // self.world_size

                module.c_proj.weight = nn.Parameter(split_tensor(module.c_proj.weight, self.world_size, dim=0)[self.rank])
                if module.c_proj.bias is not None:
                    module.c_proj.bias.data.div_(self.world_size)
                module = SequenceWarpper(module, self.group)
                setattr(upper_module, name, module)

            elif isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention):
                module.q_proj.weight = nn.Parameter(split_tensor(module.q_proj.weight, self.world_size, dim=0)[self.rank])
                if module.q_proj.bias is not None:
                    module.q_proj.bias = nn.Parameter(split_tensor(module.q_proj.bias, self.world_size, dim=0)[self.rank])
                
                module.k_proj.weight = nn.Parameter(split_tensor(module.k_proj.weight, self.world_size, dim=0)[self.rank])
                if module.k_proj.bias is not None:
                    module.k_proj.bias = nn.Parameter(split_tensor(module.k_proj.bias, self.world_size, dim=0)[self.rank])
                    
                module.v_proj.weight = nn.Parameter(split_tensor(module.v_proj.weight, self.world_size, dim=0)[self.rank])
                if module.v_proj.bias is not None:
                    module.v_proj.bias = nn.Parameter(split_tensor(module.v_proj.bias, self.world_size, dim=0)[self.rank])
                    
                module.o_proj.weight = nn.Parameter(split_tensor(module.o_proj.weight, self.world_size, dim=1)[self.rank])
                
                module.num_heads = module.num_heads // self.world_size
                module.num_key_value_heads = module.num_key_value_heads // self.world_size
                module.hidden_size = module.hidden_size // self.world_size
                module = SequenceWarpper(module, self.group)
                setattr(upper_module, name, module)
                
            elif isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2MLP):
                module.c_fc.weight = nn.Parameter(split_tensor(module.c_fc.weight, self.world_size, dim=1)[self.rank])
                if module.c_fc.bias is not None:
                    module.c_fc.bias = nn.Parameter(split_tensor(module.c_fc.bias, self.world_size, dim=0)[self.rank])
                module.c_fc.nf = module.c_fc.nf // self.world_size
                    
                module.c_proj.weight = nn.Parameter(split_tensor(module.c_proj.weight, self.world_size, dim=0)[self.rank])
                module = SequenceWarpper(module, self.group)
                setattr(upper_module, name, module)
                
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
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
    
class LinearWarpper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        group: Optional[Any] = None,
        pre = False,
        qkv = False
    ):
        super().__init__()
        self.module = module
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.pre = pre
        self.qkv = qkv
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:

        inputs = args[0]
    
        if self.pre:
            inputs = all_to_all(inputs, self.group, scatter_dim=1, gather_dim=-1)
        outputs = self.module(inputs)
        if not self.pre:
            if self.qkv:
                outputs_list = [t.contiguous() for t in torch.tensor_split(outputs, self.world_size * 3, -1)]
                outputs = torch.cat([torch.cat([outputs_list[i + j * self.world_size] for j in range(3)], dim = -1) for i in range(self.world_size)], dim = -1)
            outputs = all_to_all(outputs, self.group, scatter_dim=-1, gather_dim=1)
        return outputs


class AttentionWarpper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        group: Optional[Any] = None,
    ):
        super().__init__()
        self.module = module
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)

        if isinstance(self.module, transformers.models.gpt2.modeling_gpt2.GPT2Attention):
            self.module.num_heads = module.num_heads // self.world_size
            self.module.split_size = module.split_size // self.world_size
    
            self.replace_linear('c_attn', self.module.c_attn, qkv=True)
            self.replace_linear('c_proj', self.module.c_proj, pre=True)
        elif isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention):
            self.module.num_heads = module.num_heads // self.world_size
            self.module.num_key_value_heads = module.num_key_value_heads // self.world_size
            self.module.hidden_size = module.hidden_size // self.world_size
    
            self.replace_linear('q_proj', self.module.q_proj)
            self.replace_linear('k_proj', self.module.k_proj)
            self.replace_linear('v_proj', self.module.v_proj)
            self.replace_linear('o_proj', self.module.o_proj, pre=True)

    def replace_linear(self, name, module, pre=False, qkv=True):
        module = LinearWarpper(module, self.group, pre, qkv)
        setattr(self.module, name, module)
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        return outputs

class UlyssesParallel(nn.Module):
    def __init__(
        self,
        module,
        group: Optional[Any] = None):
        super().__init__()
        
        self.module = module
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        
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
        is_MultiheadAttention = isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention) or isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP) or isinstance(module, transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoMLP) or isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention)
        is_linear = nn.Linear

        if has_child and not is_MultiheadAttention:
            for name, child in module.named_children():
                self.RecursiveVisit(name, child, module)
        else:
            if isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention):
                module = AttentionWarpper(module, self.group)
                setattr(upper_module, name, module)
            if isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention):
                module = AttentionWarpper(module, self.group)
                setattr(upper_module, name, module)
            
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        return outputs


########################################################################


def _left_rotation(input_, buffer):

    group = torch.distributed.distributed_c10d._get_default_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
        
    send_op = torch.distributed.P2POp(torch.distributed.isend, input_, (rank + 1)%world_size)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, buffer, (rank - 1 + world_size)%world_size)

    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])

    return reqs


def _right_rotation(input_, buffer):

    group = torch.distributed.distributed_c10d._get_default_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
        
    send_op = torch.distributed.P2POp(torch.distributed.isend, input_, (rank - 1 + world_size)%world_size)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, buffer, (rank + 1)%world_size)

    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])

    return reqs
    
    
class _RotationParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, module, itr):
        return _left_rotation(input_)

    @staticmethod
    def forward(ctx, input_, module, itr):
        ctx.module = module
        ctx.itr = itr
        ctx.save_for_backward(input_)
        if itr == 0:
            module._buffer = torch.zeros_like(input_)
            module.reqs = _left_rotation(input_, module._buffer)
            return input_
        else:
            for req in module.reqs:
                req.wait()
            input_.data.copy_(module._buffer)
            if itr != torch.distributed.get_world_size() - 1:
                module.reqs = _left_rotation(input_, module._buffer)
            else:
                module._buffer = None
            return input_

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        itr = ctx.itr
        input_ = ctx.saved_tensors[0]
        if itr == torch.distributed.get_world_size() - 1:
            module._buffer = torch.zeros_like(input_)
            module.reqs = _right_rotation(input_, module._buffer)
            for req in module.reqs:
                req.wait()
            input_.data.copy_(module._buffer)
            module.reqs = _right_rotation(grad_output, module._buffer)
            for req in module.reqs:
                req.wait()
            grad_output = module._buffer
            return grad_output, None, None
        else:
            if itr != 0:
                module.reqs = _right_rotation(input_, module._buffer)
                for req in module.reqs:
                    req.wait()
                input_.data.copy_(module._buffer)
                module.reqs = _right_rotation(grad_output, module._buffer)
                for req in module.reqs:
                    req.wait()
                grad_output = module._buffer
            else:
                module._buffer = None
            return grad_output, None, None
        return grad_output, None, None


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

class GPT2Attention(nn.Module):
    def __init__(self, ref, group):
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

        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
    
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

    def rotation_generate(self, hidden_states):
        output_list = [None for _ in range(self.world_size)]
        for i in range(self.world_size):
            hidden_states = _RotationParallelRegion.apply(hidden_states, self, i)
            output_list[(i + self.rank) % self.world_size] = self.c_attn(hidden_states)
        hidden_states.buffer = None
        hidden_states.reqs = None
        
        query, key, value = torch.cat(output_list, dim=1).contiguous().split(self.split_size, dim=2)

        return query, key, value
        
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
            query, key, value = self.rotation_generate(hidden_states)

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
        attn_output = _ReduceScatterToSequenceParallelRegion.apply(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LlamaSdpaAttention(nn.Module):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def __init__(self, ref, group):
        super().__init__()
        self.config = ref.config
        self.layer_idx = ref.layer_idx
        self.attention_dropout = ref.attention_dropout
        self.hidden_size = ref.hidden_size
        self.num_heads = ref.num_heads
        self.head_dim = ref.head_dim
        self.num_key_value_heads = ref.num_key_value_heads
        self.num_key_value_groups = ref.num_key_value_groups
        self.max_position_embeddings = ref.max_position_embeddings
        self.rope_theta = ref.rope_theta
        self.is_causal = ref.is_causal

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = ref.q_proj
        self.k_proj = ref.k_proj
        self.v_proj = ref.v_proj
        self.o_proj = ref.o_proj
        self.rotary_emb = ref.rotary_emb

        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        
    def rotation_generate(self, hidden_states):
        query_list = [None for _ in range(self.world_size)]
        key_list = [None for _ in range(self.world_size)]
        value_list = [None for _ in range(self.world_size)]
        for i in range(self.world_size):
            hidden_states = _RotationParallelRegion.apply(hidden_states, self, i)
            query_list[(i + self.rank) % self.world_size] = self.q_proj(hidden_states)
            key_list[(i + self.rank) % self.world_size] = self.k_proj(hidden_states)
            value_list[(i + self.rank) % self.world_size] = self.v_proj(hidden_states)

        
        query = torch.cat(query_list, dim=1).contiguous()
        key = torch.cat(key_list, dim=1).contiguous()
        value = torch.cat(value_list, dim=1).contiguous()
        
        return query, key, value
        
    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self.rotation_generate(hidden_states)
        # query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)
        
        bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # In case static cache is used, it is an instance attribute.
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        
        attn_output = _ReduceScatterToSequenceParallelRegion.apply(attn_output)

        return attn_output, None, past_key_value

class RtpParallel(nn.Module):
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
        self.RecursiveVisit('module', self.module, self)
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
            if isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention):
                if module.c_attn is not None:
                    # print(module.c_attn.weight.shape)
                    weight_list = split_tensor(module.c_attn.weight, self.world_size * 3, dim=1)
                    my_weight_list = [weight_list[self.rank + i*self.world_size] for i in range(3)]
                    module.c_attn.weight = nn.Parameter(torch.cat(my_weight_list, dim=1).contiguous())
                    # print(module.c_attn.weight.shape)
                    
                    if module.c_attn.bias is not None:
                        bias_list = split_tensor(module.c_attn.bias, self.world_size * 3, dim=0)
                        my_bias_list = [bias_list[self.rank + i*self.world_size] for i in range(3)]
                        module.c_attn.bias = nn.Parameter(torch.cat(my_bias_list, dim=0).contiguous())
                        module.c_attn.nf = module.c_attn.nf // self.world_size
                        
                if hasattr(module, 'q_attn'):
                    module.q_attn.weight = nn.Parameter(split_tensor(module.q_attn.weight, self.world_size, dim=1)[self.rank])
                    if module.q_attn.bias is not None:
                        module.q_attn.bias = nn.Parameter(split_tensor(module.q_attn.bias, self.world_size, dim=0)[self.rank])
                        module.q_attn.nf = module.q_attn.nf // self.world_size
    
                module.num_heads = module.num_heads // self.world_size
                module.split_size = module.split_size // self.world_size

                module.c_proj.weight = nn.Parameter(split_tensor(module.c_proj.weight, self.world_size, dim=0)[self.rank])
                if module.c_proj.bias is not None:
                    module.c_proj.bias.data.div_(self.world_size)
                module = GPT2Attention(module, group = self.group)
                setattr(upper_module, name, module)

    
            elif isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention):
                module.q_proj.weight = nn.Parameter(split_tensor(module.q_proj.weight, self.world_size, dim=0)[self.rank])
                if module.q_proj.bias is not None:
                    module.q_proj.bias = nn.Parameter(split_tensor(module.q_proj.bias, self.world_size, dim=0)[self.rank])
                
                module.k_proj.weight = nn.Parameter(split_tensor(module.k_proj.weight, self.world_size, dim=0)[self.rank])
                if module.k_proj.bias is not None:
                    module.k_proj.bias = nn.Parameter(split_tensor(module.k_proj.bias, self.world_size, dim=0)[self.rank])
                    
                module.v_proj.weight = nn.Parameter(split_tensor(module.v_proj.weight, self.world_size, dim=0)[self.rank])
                if module.v_proj.bias is not None:
                    module.v_proj.bias = nn.Parameter(split_tensor(module.v_proj.bias, self.world_size, dim=0)[self.rank])
                    
                module.o_proj.weight = nn.Parameter(split_tensor(module.o_proj.weight, self.world_size, dim=1)[self.rank])
                
                module.num_heads = module.num_heads // self.world_size
                module.num_key_value_heads = module.num_key_value_heads // self.world_size
                module.hidden_size = module.hidden_size // self.world_size
                
                module = LlamaSdpaAttention(module, group = self.group)
                setattr(upper_module, name, module)
                
                
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
