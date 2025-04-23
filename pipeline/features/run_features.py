import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

GLOVE_PATH = "data/glove.6B.300d.txt"
PREPROCESSED_PATH = "data/preprocessed-bios.csv"
TFIDF_OUTPUT = "data/tfidf_matrix.pkl"
EMBEDDINGS_OUTPUT = "data/bio_embeddings.npy"

def run_tfidf(df):
    print(" Running TF-IDF...")
    docs = df["bio_cleaned"].fillna("")
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vectorizer.fit_transform(docs)
    joblib.dump((X, vectorizer), TFIDF_OUTPUT)
    print(f" TF-IDF saved to {TFIDF_OUTPUT} | Shape: {X.shape}")

def load_glove(path):
    glove = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    return glove

def average_embeddings(tokens, glove, dim=300):
    vecs = [glove[t] for t in tokens if t in glove]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

def run_embeddings(df):
    print(" Running GloVe embeddings...")
    # Ensure tokens are in list form (in case they got saved as strings)
    df["bio_tokens"] = df["bio_tokens"].apply(eval if isinstance(df["bio_tokens"][0], str) else lambda x: x)
    glove = load_glove(GLOVE_PATH)
    vectors = np.vstack([average_embeddings(t, glove) for t in df["bio_tokens"]])
    np.save(EMBEDDINGS_OUTPUT, vectors)
    print(f" Embeddings saved to {EMBEDDINGS_OUTPUT} | Shape: {vectors.shape}")
    return vectors, df["name"].values

def run_visualization(vectors, labels, preview=False):
    print(" Running PCA visualization...")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    if preview:
        plt.figure(figsize=(10, 7))
        for i, name in enumerate(labels):
            x, y = reduced[i]
            plt.scatter(x, y)
            plt.text(x + 0.2, y + 0.2, name, fontsize=9)
        plt.title("PCA of Artist Bio Embeddings")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    if not preview:
        os.makedirs("visuals", exist_ok=True)  # Ensure the 'visuals' directory exists
        plt.savefig("visuals/pca_artist_embeddings.png")
        print(" Saved PCA visualization to visuals/pca_artist_embeddings.png")
        return

def run_all():
    if not os.path.exists(PREPROCESSED_PATH):
        print(" Preprocessed file not found. Run Phase 2 first.")
        return

    df = pd.read_csv(PREPROCESSED_PATH)
    run_tfidf(df)
    vectors, labels = run_embeddings(df)
    run_visualization(vectors, labels)
    print(" All features extracted successfully!")

if __name__ == "__main__":
    run_all()
