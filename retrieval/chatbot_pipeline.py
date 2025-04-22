import atexit
from build_faiss_index import run as build_index
from run_chatbot import chat

def cleanup_resources():
    try:
        from query_retriever import cleanup_resources as cleanup_faiss
        cleanup_faiss()
    except:
        pass
    
    try:
        from generate_answer import cleanup_model
        cleanup_model()
    except:
        pass

def run_pipeline():
    try:
        print("[Phase 6] Building FAISS index...")
        build_index()
        
        print("[Phase 6] Launching chatbot...")
        chat()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cleanup_resources()

if __name__ == "__main__":
    atexit.register(cleanup_resources)
    run_pipeline()