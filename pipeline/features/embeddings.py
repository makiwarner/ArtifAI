import pandas as pd
import numpy as np
import os

GLOVE_PATH = "data/glove.6B.300d.txt" 
INPUT_PATH = "data/preprocessed-bios.csv"
OUTPUT_PATH = "data/bio_embeddings.npy"

def load_glove(path):
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    return embeddings

def average_embeddings(tokens, glove):
    vecs = [glove[token] for token in tokens if token in glove]
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(100)

def run_embeddings():
    df = pd.read_csv(INPUT_PATH)
    df["bio_tokens"] = df["bio_tokens"].apply(eval)  # convert str to list

    glove = load_glove(GLOVE_PATH)
    print(" GloVe loaded")

    embeddings = np.vstack([
        average_embeddings(tokens, glove)
        for tokens in df["bio_tokens"]
    ])

    np.save(OUTPUT_PATH, embeddings)
    print(f"[✓] Bio embeddings saved to {OUTPUT_PATH} | shape = {embeddings.shape}")

if __name__ == "__main__":
    run_embeddings()
