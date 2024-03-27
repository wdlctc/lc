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
    num_samples = 100  # Number of random samples you want to generate
    max_length = 512  # Maximum length of the sequence
    dataset = RandomDataGenerator(tokenizer, num_samples, max_length)
    
    # DataLoader
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
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
        "Peak allocated bytes on {:4f}GB".format(
            torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] / 2**30
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="openai-community/gpt2"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="yelp_review_full"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    args = parser.parse_args()

    main(args)