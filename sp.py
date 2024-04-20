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

import copy

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
                
            elif isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP):
                module.up_proj.weight = nn.Parameter(split_tensor(module.up_proj.weight, self.world_size, dim=0)[self.rank])
                    
                module.gate_proj.weight = nn.Parameter(split_tensor(module.gate_proj.weight, self.world_size, dim=0)[self.rank])
                
                module.down_proj.weight = nn.Parameter(split_tensor(module.down_proj.weight, self.world_size, dim=1)[self.rank])

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

        return outputs.contiguous()


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

    def replace_linear(self, name, module, pre=False, qkv=False):
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


def _left_rotation(input_, buffer, i):

    group = torch.distributed.distributed_c10d._get_default_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
        
    send_op = torch.distributed.P2POp(torch.distributed.isend, input_, (rank + i)%world_size)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, buffer, (rank - i + world_size)%world_size)

    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])

    return reqs


def _right_rotation(input_, buffer, i):

    group = torch.distributed.distributed_c10d._get_default_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
        
    send_op = torch.distributed.P2POp(torch.distributed.isend, input_, (rank - i + world_size)%world_size)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, buffer, (rank + i)%world_size)

    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])

    return reqs


class _RotationParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, module, itr, index):
        return _left_rotation(input_)

    @staticmethod
    def forward(ctx, input_, module, itr, index):
        ctx.module = module
        ctx.itr = itr
        # ctx.save_for_backward(input_)
        if itr == 0:
            module._buffer = torch.zeros_like(module.flat_param)
            module.reqs = _left_rotation(module.flat_param.data, module._buffer, 1)
            return input_
        else:
            for req in module.reqs:
                req.wait()
            module.flat_param.data.copy_(module._buffer)
            if itr != torch.distributed.get_world_size() - 1:
                module.reqs = _left_rotation(module.flat_param.data, module._buffer, 1)
            else:
                module._buffer = None
            return input_

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        itr = ctx.itr
        # input_ = ctx.saved_tensors[0]
        if itr == torch.distributed.get_world_size() - 1:
            module._buffer = torch.zeros_like(module.flat_param)
            module._grad_buffer = torch.zeros_like(module.flat_param)
            module.reqs = _right_rotation(module.flat_param.data, module._buffer, 1)
            module.grad_reqs = _right_rotation(module._full_grad.data, module._grad_buffer, 1)
            for req in module.reqs:
                req.wait()
            for req in module.grad_reqs:
                req.wait()
            module.flat_param.data.copy_(module._buffer)
            module._full_grad.data.copy_(module._grad_buffer)
            return grad_output, None, None, None
        else:
            if itr != 0:
                module.reqs = _right_rotation(module.flat_param.data, module._buffer, 1)
                module.grad_reqs = _right_rotation(module._full_grad.data, module._grad_buffer, 1)
                for req in module.reqs:
                    req.wait()
                for req in module.grad_reqs:
                    req.wait()
                module.flat_param.data.copy_(module._buffer)
                module._full_grad.data.copy_(module._grad_buffer)
            else:
                module._buffer = None
                module._grad_buffer = None
            return grad_output, None, None, None

class _RotationParallelRegion_all(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(ctx, input_, buffer, module, itr, index):
        return _left_rotation(input_)

    @staticmethod
    def forward(ctx, input_, buffer, module, itr, index):
        bsz, q_len, _ = input_.size()
        
        ctx.module = module
        ctx.itr = itr
        ctx.q_len = q_len
        ctx.index = index
        
        if itr == 0:
            buffer[:, q_len*index:q_len*(index + 1), :] = input_
        else:
            module.all_reqs = _right_rotation(input_, buffer[:, q_len*index:q_len*(index + 1), :] , itr)
            if itr == torch.distributed.get_world_size() - 1:
                for req in module.all_reqs:
                    req.wait()
        return buffer

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        itr = ctx.itr
        index = ctx.index
        q_len = ctx.q_len

        group = torch.distributed.distributed_c10d._get_default_group()
        rank = torch.distributed.get_rank(group)
        world_size = torch.distributed.get_world_size(group)

        print(itr)
        
        if itr == torch.distributed.get_world_size() - 1:
            module._full_grad = torch.zeros_like(module.flat_param)
            for sub_module in module.module_list:
                cur_numel = 0
                for param in sub_module.parameters():
                    param.grad = module._full_grad[cur_numel: cur_numel + param.numel()].view(param.shape)
                    cur_numel += param.numel()

        if itr == torch.distributed.get_world_size() - 1:
            grad_all = grad_output.clone().detach()
            req_list = [None for _ in range(world_size)]
            for i in range(world_size-1, 0, -1):
                if i == 0:
                    continue
                
                index = (i + rank) % world_size
                buffer_ = grad_output[:, q_len*index:q_len*(index + 1), :]
                grad = grad_all[:, q_len*index:q_len*(index + 1), :]
                reqs = _left_rotation(grad, buffer_, i) 
                req_list[i] = reqs
            module.req_list = req_list
            
        index = ctx.index
        
        if itr != 0:
            for req in module.req_list[i]:
                req.wait()
                
        grad = grad_output[:, q_len*index:q_len*(index + 1), :]
        return grad, grad_output, None, None, None

class RtpLinearWarpper(nn.Module):
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
        self.out_features = self.module.out_features

        param_list = list(module.parameters())

        self._param_numels = [p.numel() for p in param_list]

        self.flat_param = torch.cat([p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1) for p in param_list], 0)
        # self.flat_param = nn.Parameter(self.flat_param, requires_grad=param_list[0].requires_grad)

        cur_numel = 0
        for param in param_list:
            param.data = self.flat_param[cur_numel: cur_numel + param.numel()].view(param.shape)
            cur_numel += param.numel()

        self.module_list = []
        for i in range(self.world_size):
            if i == self.rank:
                sub_module = self.module
                self.module_list.append(sub_module)
                continue
            sub_module = copy.deepcopy(self.module)

            cur_numel = 0
            for param in sub_module.parameters():
                param.data = self.flat_param[cur_numel: cur_numel + param.numel()].view(param.shape)
                cur_numel += param.numel()

            self.module_list.append(sub_module)
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        output_list = [None for _ in range(self.world_size)]

        

        if self.pre:
            inputs = all_to_all(inputs, self.group, scatter_dim=1, gather_dim=-1)
            for i in range(self.world_size):
                index = (self.rank + i +self.world_size) % self.world_size
                inputs = _RotationParallelRegion.apply(inputs, self, i, index)
                outputs = self.module_list[index](inputs, **kwargs)
                output_list[index] = outputs
        else:
            inputs = args[0]
            bsz, q_len, _ = inputs.size()
            outputs_buffer = torch.zeros((bsz, q_len * self.world_size, self.out_features // self.world_size), device=inputs.device, dtype=inputs.dtype)
            for i in range(self.world_size):
                index = (self.rank + i +self.world_size) % self.world_size
                inputs = _RotationParallelRegion.apply(inputs, self, i, index)
                outputs = self.module_list[index](inputs, **kwargs)
                outputs_buffer = _RotationParallelRegion_all.apply(outputs, outputs_buffer, self, i, index)

        return outputs_buffer


class RtpWarpper(nn.Module):
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
            
            self.module.q_proj.weight = nn.Parameter(split_tensor(self.module.q_proj.weight, self.world_size, dim=0)[self.rank])
            if self.module.q_proj.bias is not None:
                self.module.q_proj.bias = nn.Parameter(split_tensor(self.module.q_proj.bias, self.world_size, dim=0)[self.rank])
            
            self.module.k_proj.weight = nn.Parameter(split_tensor(self.module.k_proj.weight, self.world_size, dim=0)[self.rank])
            if self.module.k_proj.bias is not None:
                self.module.k_proj.bias = nn.Parameter(split_tensor(self.module.k_proj.bias, self.world_size, dim=0)[self.rank])
                
            self.module.v_proj.weight = nn.Parameter(split_tensor(self.module.v_proj.weight, self.world_size, dim=0)[self.rank])
            if self.module.v_proj.bias is not None:
                self.module.v_proj.bias = nn.Parameter(split_tensor(self.module.v_proj.bias, self.world_size, dim=0)[self.rank])
                
            # self.module.o_proj.weight = nn.Parameter(split_tensor(self.module.o_proj.weight, self.world_size, dim=0)[self.rank])
            # if self.module.o_proj.bias is not None:
            #     self.module.o_proj.bias = nn.Parameter(split_tensor(self.module.o_proj.bias, self.world_size, dim=0)[self.rank])
                
            self.module.num_heads = module.num_heads // self.world_size
            self.module.num_key_value_heads = module.num_key_value_heads // self.world_size
            self.module.hidden_size = module.hidden_size // self.world_size
    
            self.replace_linear('q_proj', self.module.q_proj)
            self.replace_linear('k_proj', self.module.k_proj)
            self.replace_linear('v_proj', self.module.v_proj)
            self.replace_linear_u('o_proj', self.module.o_proj, pre=True)

    
    def replace_linear_u(self, name, module, pre=False, qkv=False):
        module = LinearWarpper(module, self.group, pre, qkv)
        setattr(self.module, name, module)

    def replace_linear(self, name, module, pre=False, qkv=True):
        module = RtpLinearWarpper(module, self.group, pre, qkv)
        setattr(self.module, name, module)
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        return outputs

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
            if isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention):
                
                module = RtpWarpper(module, self.group)
                setattr(upper_module, name, module)
            if isinstance(module, transformers.models.gpt2.modeling_gpt2.GPT2Attention):
                module = RtpWarpper(module, self.group)
                setattr(upper_module, name, module)
                
            elif isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP):
                module.up_proj.weight = nn.Parameter(split_tensor(module.up_proj.weight, self.world_size, dim=0)[self.rank])
                    
                module.gate_proj.weight = nn.Parameter(split_tensor(module.gate_proj.weight, self.world_size, dim=0)[self.rank])
                
                module.down_proj.weight = nn.Parameter(split_tensor(module.down_proj.weight, self.world_size, dim=1)[self.rank])

                module = RtpWarpper(module, self.group)
                setattr(upper_module, name, module)
                
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        # if not self.eval:
        #     self._register_post_backward_hooks()
        return outputs

    # def _register_post_backward_hooks(self) -> None:
    #     if not torch.is_grad_enabled():
    #         return  # don't register grad hooks if grad isn't enabled
    #     for module in self.FlyweightModule_list:
    #         for i, sub_module in enumerate(module.module_list):
    #             if i == self.rank:
    #                 continue
    #             sub_module.count = 0
    #             for p in sub_module.parameters():
    #                 if p.requires_grad:
    #                     # Register a hook.
    #                     p_tmp = p.expand_as(p)  # Get a grad_fn on p_tmp.
    #                     assert p_tmp.grad_fn is not None
    #                     grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.
    #                     sub_module.count += 1

    #                     if  not hasattr(p, '_my_handle'):
    #                         handle = grad_acc.register_hook(partial(hook_fn, sub_module, module))
    #                         p._my_handle = handle

