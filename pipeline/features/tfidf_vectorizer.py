import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

DATA_PATH = "data/preprocessed-bios.csv"
OUTPUT_PATH = "data/tfidf_matrix.pkl"

def run_tfidf():
    df = pd.read_csv(DATA_PATH)
    docs = df["bio_cleaned"].fillna("")

    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X = vectorizer.fit_transform(docs)

    joblib.dump((X, vectorizer), OUTPUT_PATH)
    print(f" TF-IDF matrix saved to {OUTPUT_PATH} | shape = {X.shape}")

if __name__ == "__main__":
    run_tfidf()
