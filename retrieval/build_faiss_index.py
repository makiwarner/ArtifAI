import numpy as np
import pandas as pd
import faiss
import os
from sentence_transformers import SentenceTransformer

# === CONFIG ===
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "data/new_bio_embeddings.npy"
ARTIST_DATA_PATH = "data/preprocessed-bios.csv"
INDEX_OUTPUT = "retrieval/faiss_index.index"
NAMES_OUTPUT = "retrieval/artist_names.txt"

def run():
    print("[Phase 6] Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    expected_dim = model.get_sentence_embedding_dimension()
    print(" Embedding dimension from model:", expected_dim)

    print("[Phase 6] Loading artist embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f" Embedding dimension mismatch! Expected {expected_dim}, "
            f"but got {embeddings.shape[1]}"
        )

    df = pd.read_csv(ARTIST_DATA_PATH)
    artist_names = df["name"].tolist()

    print("[Phase 6] Building FAISS index...")
    index = faiss.IndexFlatL2(expected_dim)
    index.add(embeddings)

    # Save outputs
    faiss.write_index(index, INDEX_OUTPUT)
    with open(NAMES_OUTPUT, "w") as f:
        for name in artist_names:
            f.write(name.strip() + "\n")

    print(f" FAISS index saved to {INDEX_OUTPUT}")
    print(f" Artist names saved to {NAMES_OUTPUT}")

if __name__ == "__main__":
    run()
