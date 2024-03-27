import os
import torch
from megatron.core import parallel_state
import time

def initialize_distributed(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1):
    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

def model_provider():
    """Build the model."""

    # GPT-2 configuration - Adjusted for a generic GPT-2-like model setup
    transformer_config = TransformerConfig(
        num_layers=12,  # GPT-2 "small" has 12 layers. Adjust accordingly for larger variants
        hidden_size=768,  # GPT-2 "small" has a hidden size of 768. Adjust for larger variants
        num_attention_heads=12,  # GPT-2 "small" uses 12 attention heads. Adjust accordingly
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,)  # Epsilon used for layer normalization

    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=100,
        max_sequence_length=512)

    return gpt_model

from torch.utils.data import DataLoader
from megatron.core.datasets.utils import Split
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset

def get_train_data_iterator():
    config = GPTDatasetConfig(
        is_built_on_rank=lambda:(parallel_state.is_pipeline_last_stage() or parallel_state.is_pipeline_first_stage()),
        random_seed = 0,
        sequence_length = 512,
        blend=[],
        mock=True,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer="dummy")

    training_data= MockGPTDataset(Split.train, config)

    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)

    train_iterator = iter(train_dataloader)
    return train_iterator

from functools import partial

def forward_step_func(data_iterator, model):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    data = next(data_iterator)
    tokens = data['tokens'].to(device)
    attention_mask = data['attention_mask'].to(device)
    position_ids = data['position_ids'].to(device)
    labels = data['labels'].to(device)
    loss_mask = data['loss_mask'].to(device)

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)

from megatron.core import dist_checkpointing

from pathlib import Path
from torch.optim import Adam
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

if __name__ == "__main__":
    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    gpt_model = model_provider()
    device = torch.device("cuda")
    gpt_model.to(device)

    optim = Adam(gpt_model.parameters())

    train_iterator = get_train_data_iterator()

    forward_backward_func = get_forward_backward_func()

    # Running the model for 5 iterations
    for _ in range(5):
        start_time = time.time()
        optim.zero_grad()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=100,
            seq_length=512,
            micro_batch_size=1,
            decoder_seq_length=512,
            forward_only=False)

        optim.step()

        print(f'Losses reduced :{len(losses_reduced)}')
        epoch_time = time.time() - start_time
        print(f"Time: {epoch_time:.2f} seconds")

    print(
        "Peak allocated bytes on cuda:{}: {:4f}GB".format(
            int(os.environ['LOCAL_RANK']), torch.cuda.memory_stats(int(os.environ['LOCAL_RANK']))["allocated_bytes.all.peak"] / 2**30
        )
    )
