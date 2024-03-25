import argparse
import os
import time

import torch

from utils import load, load_jsonl, load_data
from datasets import load_dataset, load_from_disk

from transformers import TrainingArguments

def inference(model, tokenizer, prompts, max_length=50):
    for idx, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            output = model.generate(input_ids, max_length=100, num_return_sequences=1)
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        print("Generated text:", generated_text)

def main(args):
    # Specify the pretrained model name or path
    model_name = args.model_name
    
    # Load the tokenizer and pretrained model
    model, tokenizer = load(model_name)

    # dataset = load_data(args.dataset_name, tokenizer)
    # dataset.save_to_disk(args.dataset_name)

    dataset = load_from_disk(args.dataset_name)
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
    )

    trainer.train()
    
    # Move the model to GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")
    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    start_time = time.time()
    inference(model, tokenizer, prompts)
    generation_time = time.time() - start_time
    print(f"Text generation completed in {generation_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="yelp_review_full"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    args = parser.parse_args()

    main(args)