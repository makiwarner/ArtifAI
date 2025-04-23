import os
import subprocess

def log(msg):
    print(f"\n[Phase 4] {msg}")

def run_formatting():
    log("Generating prompt-response training data...")
    from format_training_data import run as format_data
    format_data()
    log(" Training data generated.")

def run_training():
    log("Starting model fine-tuning...")
    from train_dialogue_model import trainer, model, tokenizer, OUTPUT_DIR

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log(" Model and tokenizer saved to output/")

if __name__ == "__main__":
    run_formatting()
    run_training()
