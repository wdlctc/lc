import argparse
import os
import time

import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

from torch.optim import AdamW

from utils import load, load_jsonl, load_data
from datasets import load_dataset, load_from_disk

from transformers import TrainingArguments, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM

import numpy as np
import datasets
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        worker_info = None
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}
       
def collate_fn(batch_list):
    batch = {
        "input_ids": torch.stack([torch.Tensor(example["input_ids"]).long() for example in batch_list]),
        "attention_mask": torch.stack([torch.Tensor(example["attention_mask"]).long() for example in batch_list]),
    }
    return batch

def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch
        
def evaluate_model(model, preprocess_batched, pad_idx, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True)
    val_data = val_data.shuffle(seed=42)
    print(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    print(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item()

    total_loss = total_loss / total_batches

    return total_loss, evaluated_on_tokens 

def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def main(args):
    # Specify the pretrained model name or path
    model_name = args.model_name
    
    # Load the tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
    model_config = AutoConfig.from_pretrained(args.model_config)

    init_random_seed(42)
    
    model = LlamaForCausalLM(model_config)
    tokenizer.pad_token = tokenizer.eos_token
    pad_idx = tokenizer.pad_token_id
    model = model.bfloat16()
    
    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.gradient_checkpointing_enable()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-5, weight_decay=args.weight_decay)
    
    # Instantiate the dataset
    num_samples = args.num_samples  # Number of random samples you want to generate
    max_length = args.max_length  # Maximum length of the sequence

    # Load the "allenai/c4" dataset with streaming=True
    
    # dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    dataset = load_dataset('togethercomputer/Long-Data-Collections', "default", split="train", streaming=True)
    dataset = dataset['train']
    # dataset = load_dataset("togethercomputer/Long-Data-Collections", "default", split="train", streaming=True)
    # dataset = PreprocessedIterableDataset(dataset, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch
        
    tokenized_datasets = dataset.map(lambda x: preprocess_batched(x), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "prompt", "completion"])

    def convert_to_torch_dataset(tokenized_dataset):
        return tokenized_dataset.with_format("torch")
    
    tokenized_datasets = convert_to_torch_dataset(tokenized_datasets)
    
    step = 0
    num_epochs = 3
    log_interval = 10
    update_time = time.time()
    total_loss = 0
    losses = []

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"output_{max_length}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_total_limit=2,
        weight_decay=0.01,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
    )

    # Train the model
    trainer.train()
    
    # init_random_seed(42)
    # for epoch in range(num_epochs):
    #     model.train()
    #     for batch in dataloader:

    #         step += 1

    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         labels = batch["input_ids"].clone()
    #         labels[labels == pad_idx] = -100
            
    #         loss = model(**batch, labels=labels).loss
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
    #         optimizer.step()
    #         optimizer.zero_grad()

    #         total_loss += loss
    #         if step % log_interval == 0:
    #             print(total_loss.item() / log_interval)
    #             losses.append(total_loss.item()/ log_interval)
    #             total_loss = 0
                
    #     break
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    
    # output_dir = "trained_model"
    # os.makedirs(output_dir, exist_ok=True)
    
    # model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)


    # batch_size=args.batch_size
    # _time = time.time()
    # val_data = load_dataset("togethercomputer/Long-Data-Collections", "default", split="train", streaming=True)
    # val_data = val_data.shuffle(seed=42)
    # print(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    # val_data_mapped = val_data.map(
    #     preprocess_batched,
    #     batched=True,
    #     remove_columns=["text"],
    # )
    # val_data_mapped.batch = lambda batch_size: batch_fn(val_data_mapped, batch_size)

    # target_eval_tokens = 10_000_000
    # evaluated_on_tokens = 0
    # total_loss = torch.tensor(0.0).to(device)
    # total_batches = 1
    # print(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    # for batch in val_data_mapped.batch(batch_size=batch_size):
    #     if evaluated_on_tokens > target_eval_tokens:
    #         break
    #     total_batches += 1

    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     labels = batch["input_ids"].clone()
    #     labels[labels == pad_idx] = -100
    #     loss = model(**batch, labels=labels).loss
    #     total_loss += loss.detach()

    #     evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item()

    # total_loss = total_loss / total_batches

    # print(total_loss)

    # import csv
    # output_file = "training_loss.csv"
    # with open(output_file, "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Step", "Loss"])
    #     for i, loss in enumerate(losses):
    #         writer.writerow([i, loss])

    print(
        "Peak allocated bytes on {:4f}GB".format(
            torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] / 2**30
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config", type=str, default="configs/llama_60m.json"
    )
    parser.add_argument(
        "--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--num_samples", type=int, default=10
    )
    parser.add_argument(
        "--max_length", type=int, default=8192
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--grad_clipping", type=float, default=1)
    args = parser.parse_args()

    main(args)
