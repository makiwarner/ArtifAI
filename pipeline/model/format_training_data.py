import pandas as pd
import json
import os

INPUT_PATH = "data/preprocessed-bios.csv"
OUTPUT_PATH = "data/train_data.jsonl"

TEMPLATES = [
    lambda name: f"Who was {name}?",
    lambda name: f"What is important to know about {name}?",
    lambda name: f"Can you tell me about the life and work of {name}?",
    lambda name: f"What made {name} a notable artist?",
    lambda name: f"Give me a brief biography of {name}."
]

def generate_prompt_response(name, bio):
    return [
        {"prompt": template(name), "response": bio}
        for template in TEMPLATES
    ]

def run():
    df = pd.read_csv(INPUT_PATH)
    data = []
    for _, row in df.iterrows():
        entries = generate_prompt_response(row["name"], row["bio_cleaned"])
        data.extend(entries)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

if __name__ == "__main__":
    run()
