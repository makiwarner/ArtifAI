from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.schema import Document
import os

# Load FAISS retriever with cosine similarity logic
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Updated for langchain_community

def load_retriever(index_path="embeddings/faiss_index", artist_dir="artists"):
    from scripts.build_index import build_faiss_index
    if not os.path.exists(index_path):
        print(" FAISS index not found. Building it now...")
        build_faiss_index(artist_dir, index_path)

    db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.6, "k": 5}
    )
    return retriever