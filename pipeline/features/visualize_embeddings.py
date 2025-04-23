import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

INPUT_PATH = "data/bio_embeddings.npy"
LABELS_PATH = "data/preprocessed-bios.csv"

def visualize(method="pca"):
    X = np.load(INPUT_PATH)
    df = pd.read_csv(LABELS_PATH)
    labels = df["name"].values

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=5, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    X_reduced = reducer.fit_transform(X)

    plt.figure(figsize=(12, 8))
    for i, label in enumerate(labels):
        x, y = X_reduced[i]
        plt.scatter(x, y)
        plt.text(x + 0.2, y + 0.2, label, fontsize=9)
    plt.title(f"{method.upper()} Projection of Artist Bios")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize("pca")  # or "tsne"
