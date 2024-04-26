import argparse
import os
import time

import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

from torch.optim import AdamW

from utils import load, load_jsonl, load_data
from datasets import load_dataset, load_from_disk

from transformers import TrainingArguments, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM

import numpy as np
import datasets
from torch.utils.data import IterableDataset, DataLoader


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

def main(args):
    # Specify the pretrained model name or path
    model_name = args.model_name
    
    # Load the tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
    model_config = AutoConfig.from_pretrained(args.model_config)
    model = LlamaForCausalLM(model_config)
    tokenizer.pad_token = tokenizer.eos_token
    pad_idx = tokenizer.pad_token_id
    
    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Instantiate the dataset
    num_samples = args.num_samples  # Number of random samples you want to generate
    max_length = args.max_length  # Maximum length of the sequence

    # Load the "allenai/c4" dataset with streaming=True
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = PreprocessedIterableDataset(dataset, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    step = 0
    num_epochs = 3

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
        
    mini_sequence = 4
    position_ids_list, mask_list = get_mini_sequence_list(args.max_length, mini_sequence)
    sub_length = args.max_length // mini_sequence
    
    update_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:

            step += 1
            
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = batch["input_ids"]
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100

            inputs_list = []
            label_list = []
            attention_mask_list = []

            for i in range(mini_sequence):
                inputs_list.append(inputs[:, i*(sub_length): (i+1)*(sub_length)])

            for i in range(mini_sequence):
                if i != mini_sequence - 1:
                    label_list.append(labels[:, 1 + i*(sub_length): 1 + (i+1)*(sub_length)])
                else:
                    label_list.append(labels[:, 1 + i*(sub_length): (i+1)*(sub_length)])
                    label_list[i].end = True
                label_list[i].mini = True

            
            for i in range(mini_sequence):
                inputs_list.append(inputs[:, i*(sub_length): (i+1)*(sub_length)])
    
            for i in range(mini_sequence):
                attention_mask_list.append(batch["attention_mask"][:, : (i+1)*(sub_length)])
                
            past_key_values = None
            for i in range(mini_sequence):
                outputs = model(input_ids=inputs_list[i], labels=label_list[i], position_ids=position_ids_list[i], attention_mask = attention_mask_list[i], use_cache=True, past_key_values=past_key_values)
                loss = outputs.loss
                loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1/mini_sequence)
            optimizer.step()
            optimizer.zero_grad()

            print(loss)
            if step == 100:
                return
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config", type=str, default="configs/llama_60m.json"
    )
    parser.add_argument(
        "--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16
    )
    parser.add_argument(
        "--num_samples", type=int, default=10
    )
    parser.add_argument(
        "--max_length", type=int, default=512
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=100)
    args = parser.parse_args()

    main(args)