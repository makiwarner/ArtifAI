import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import os

# Using a model that produces 300-dimensional embeddings to match bio_embeddings.npy
EMBED_MODEL_NAME = "sentence-transformers/distilbert-base-nli-mean-tokens"
EMBEDDINGS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "bio_embeddings.npy"))
ARTIST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed-bios.csv"))

# Initialize the model and load data
model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
embeddings = np.load(EMBEDDINGS_PATH)
artist_names = pd.read_csv(ARTIST_DATA_PATH)["name"].tolist()

# Build nearest neighbor model
nn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
nn_model.fit(embeddings)

def query_artists(question, k=3):
    try:
        print(f"Processing question: {question}")
        
        # Direct name matching for questions that mention artist names
        artist_names_lower = [name.lower() for name in artist_names]
        question_lower = question.lower().split()
        
        # Check for direct name mentions first
        direct_matches = []
        for idx, name in enumerate(artist_names_lower):
            name_parts = name.split()
            # Match if full name or last name is mentioned
            if any(part in question_lower for part in name_parts):
                direct_matches.append((idx, 1.0))
        
        if direct_matches:
            results = [(artist_names[idx], score) for idx, score in direct_matches]
            print(f"Found {len(results)} matching artists through direct name match")
            return results[:k]
            
        # If no direct matches, use semantic search
        query_vec = model.encode([question])
        distances, indices = nn_model.kneighbors(query_vec, n_neighbors=min(k, len(artist_names)))
        
        if len(indices[0]) == 0:
            print("No matching artists found")
            return []
            
        results = [(artist_names[i], float(1 - dist)) for i, dist in zip(indices[0], distances[0])]
        print(f"Found {len(results)} matching artists through semantic search")
        return results
        
    except Exception as e:
        print(f"Error in query_artists: {str(e)}")
        return []

if __name__ == "__main__":
    prompt = input("Ask something about an artist: ")
    top_matches = query_artists(prompt)
    print("\nTop matches:")
    for name, score in top_matches:
        print(f"- {name} (score={score:.4f})")
    print(" Embedding dimension:", model.get_sentence_embedding_dimension())
