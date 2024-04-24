import argparse
import os
import time

import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

from torch.optim import AdamW

from utils import load, load_jsonl, load_data
from datasets import load_dataset, load_from_disk

from transformers import TrainingArguments, TextDataset, DataCollatorForLanguageModeling

import numpy as np
import copy

def main(args):
    # Specify the pretrained model name or path
    model_name = args.model_name
    
    # Load the tokenizer and pretrained model
    model, tokenizer = load(model_name)
    
    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
    
    # DataLoader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Set up the optimizer
    # Training loop
    num_epochs = 3
    mini_sequence = 4

    position_ids_list = []
    sub_length = args.max_length // mini_sequence
    for i in range(mini_sequence):
        position_ids_list.append(torch.arange(
            sub_length * i, sub_length * (i+1), device=device
        ).unsqueeze(0))

    mask_list = []
    
    # temp_mask = torch.ones(sub_length, sub_length, dtype=torch.bool).tril(diagonal=0).cuda().unsqueeze(0).unsqueeze(0)
    # full_mask = torch.ones(sub_length, sub_length, dtype=torch.bool).cuda().unsqueeze(0).unsqueeze(0)
    # all_mask = temp_mask
    
    for i in range(mini_sequence):
        mask_list.append(None)
        
    #     mask_list.append(all_mask)
    #     all_mask = torch.cat((full_mask, all_mask), dim=-1)

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
    
        for batch in data_loader:
            inputs = batch.to(device)

            inputs_list = []
            label_list = []

            for i in range(mini_sequence):
                inputs_list.append(inputs[:, i*(sub_length): (i+1)*(sub_length)])

            for i in range(mini_sequence):
                if i != mini_sequence - 1:
                    label_list.append(inputs[:, 1 + i*(sub_length): 1 + (i+1)*(sub_length)])
                else:
                    label_list.append(inputs[:, 1 + i*(sub_length): (i+1)*(sub_length)])
                    label_list[i].end = True
                label_list[i].mini = True

            past_key_values = None
            for i in range(mini_sequence):
                if past_key_values is not None:
                    past_key_values = list(past_key_values)
                    for layer_idx in range(len(past_key_values)):
                        key_states, value_states = past_key_values[layer_idx]
                        past_key_values[layer_idx] = (key_states.detach(), value_states.detach())
                    past_key_values = tuple(past_key_values)

                outputs = model(input_ids=inputs_list[i], labels=label_list[i], position_ids=position_ids_list[i], attention_mask = mask_list[i], use_cache=True, past_key_values=past_key_values)
                loss = outputs.loss
                past_key_values = outputs.past_key_values
                loss.backward()
                
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
    
        avg_loss = total_loss / len(data_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f} - Time: {epoch_time:.2f} seconds")


    print(
        "Peak allocated bytes on {:4f}GB".format(
            torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] / 2**30
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
        "--max_length", type=int, default=512
    )
    parser.add_argument("--data_root", type=str, default="data/")
    args = parser.parse_args()

    main(args)