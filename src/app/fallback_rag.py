from langchain_community.document_loaders import WikipediaLoader

def fetch_from_wikipedia(query, lang="en"):
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=2)
    docs = loader.load()
    return docs