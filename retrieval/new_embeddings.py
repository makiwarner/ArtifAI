import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

INPUT_PATH = "data/preprocessed-bios.csv"
OUTPUT_PATH = "data/new_bio_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def run():
    print("[Phase 3] Loading bios...")
    df = pd.read_csv(INPUT_PATH)
    texts = df["bio_cleaned"].fillna("").tolist()

    print(f"[Phase 3] Encoding {len(texts)} bios using {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, convert_to_numpy=True)

    os.makedirs("data", exist_ok=True)
    np.save(OUTPUT_PATH, embeddings)
    print(f" Embeddings saved to {OUTPUT_PATH}")
    print(f" Shape: {embeddings.shape}")

if __name__ == "__main__":
    run()
