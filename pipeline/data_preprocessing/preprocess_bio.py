import pandas as pd
import os
from clean_text import clean_text
from tokenize_lemmatize import tokenize_lemmatize
from extract_entities import extract_entities

INPUT_PATH = "data/best-artworks-updated.csv"
OUTPUT_PATH = "data/preprocessed-bios.csv"

def preprocess():
    df = pd.read_csv(INPUT_PATH)

    cleaned_bios = []
    tokens = []
    entities = []

    for bio in df["bio"]:
        cleaned = clean_text(str(bio))
        cleaned_bios.append(cleaned)
        tokens.append(tokenize_lemmatize(cleaned))
        entities.append(extract_entities(cleaned))

    df["bio_cleaned"] = cleaned_bios
    df["bio_tokens"] = tokens
    df["bio_entities"] = entities

    df.to_csv(OUTPUT_PATH, index=False)
    print(f" Preprocessed bios saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()
