from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
import torch

MODEL_NAME = "gpt2"
DATA_PATH = "data/train_data.jsonl"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    # Ensure 'prompt' and 'response' are strings before concatenation
    prompt = " ".join(example['prompt']) if isinstance(example['prompt'], list) else example['prompt']
    response = " ".join(example['response']) if isinstance(example['response'], list) else example['response']
    return tokenizer(prompt + "\n" + response, truncation=True, padding="max_length", max_length=250)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
tokenized_dataset = dataset.map(tokenize, batched=False)

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    warmup_steps=10,
    save_total_limit=2,
    evaluation_strategy="no",
    #fp16=torch.cuda.is_available() uncomment if you want to use mixed precision training with GPU
    fp16=False,
    overwrite_output_dir=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)