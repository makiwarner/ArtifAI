import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Get the absolute path to the data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

docs = []
artist_dir = DATA_DIR

for file in os.listdir(artist_dir):
    if file.endswith(".json"):
        artist_name = file.replace(".json", "")
        file_path = os.path.join(artist_dir, file)
        try:
            with open(file_path, "r") as f:
                if os.stat(file_path).st_size == 0:  # Skip empty files
                    print(f"Skipping empty file: {file}")
                    continue
                data = json.load(f)  # Attempt to load JSON
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON file: {file}")
            continue

        for section, content in data.items():
            if isinstance(content, list):
                content = " ".join(
                    [json.dumps(item) if isinstance(item, dict) else str(item) for item in content]
                )
            elif isinstance(content, dict):
                content = json.dumps(content)
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                docs.append(Document(page_content=chunk, metadata={"artist": artist_name, "section": section}))

db = FAISS.from_documents(docs, embedding_model)
db.save_local("embeddings/faiss_index")
print(" FAISS index built successfully.")


def build_faiss_index(artist_dir="artists", index_path="embeddings/faiss_index"):
    from langchain.schema import Document
    from langchain.text_splitter import CharacterTextSplitter
    import json

    docs = []
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file in os.listdir(artist_dir):
        if file.endswith(".json"):
            artist_name = file.replace(".json", "")
            file_path = os.path.join(artist_dir, file)
            try:
                with open(file_path, "r") as f:
                    if os.stat(file_path).st_size == 0:  # Skip empty files
                        print(f"Skipping empty file: {file}")
                        continue
                    data = json.load(f)  # Attempt to load JSON
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {file}")
                continue

            for section, content in data.items():
                if isinstance(content, list):
                    content = " ".join(
                        [json.dumps(item) if isinstance(item, dict) else str(item) for item in content]
                    )
                elif isinstance(content, dict):
                    content = json.dumps(content)
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    docs.append(Document(page_content=chunk, metadata={"artist": artist_name, "section": section}))

    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(index_path)
    print(" FAISS index built successfully.")