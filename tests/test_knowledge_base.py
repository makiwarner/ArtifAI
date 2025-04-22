import os
import json
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from test_config import (
    DATA_DIR, REPORTS_DIR, EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD
)
from src.chatbot import ArtistChatbot

class KnowledgeBaseTester:
    def __init__(self, artist_data_path: str):
        self.chatbot = ArtistChatbot(artist_data_path)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
    def test_knowledge_coverage(self) -> Dict:
        """Test the coverage of different knowledge domains."""
        knowledge_base = self.chatbot._create_knowledge_base()
        coverage = {}
        
        for domain, content in knowledge_base.items():
            if isinstance(content, dict):
                coverage[domain] = {
                    'fields': len(content),
                    'non_empty_fields': sum(1 for v in content.values() if v),
                    'content_types': {
                        'string': sum(1 for v in content.values() if isinstance(v, str)),
                        'list': sum(1 for v in content.values() if isinstance(v, list)),
                        'dict': sum(1 for v in content.values() if isinstance(v, dict))
                    }
                }
            elif isinstance(content, list):
                coverage[domain] = {
                    'items': len(content),
                    'non_empty_items': sum(1 for item in content if item)
                }
        
        return coverage
    
    def test_embedding_quality(self) -> Dict:
        """Test the quality of embeddings for different knowledge domains."""
        knowledge_base = self.chatbot._create_knowledge_base()
        embeddings = self.chatbot._create_knowledge_embeddings()
        quality_metrics = {}
        
        for domain, content in knowledge_base.items():
            if domain in embeddings:
                domain_embeddings = embeddings[domain]
                if isinstance(domain_embeddings, dict):
                    # Calculate average similarity between embeddings in the same domain
                    similarities = []
                    embedding_list = list(domain_embeddings.values())
                    
                    for i in range(len(embedding_list)):
                        for j in range(i + 1, len(embedding_list)):
                            if isinstance(embedding_list[i], np.ndarray) and isinstance(embedding_list[j], np.ndarray):
                                similarity = cosine_similarity([embedding_list[i]], [embedding_list[j]])[0][0]
                                similarities.append(float(similarity))  # Convert to Python float
                    
                    quality_metrics[domain] = {
                        'avg_similarity': float(np.mean(similarities)) if similarities else 0.0,  # Convert to Python float
                        'std_similarity': float(np.std(similarities)) if similarities else 0.0,  # Convert to Python float
                        'embedding_count': len(embedding_list),
                        'high_similarity_pairs': sum(1 for s in similarities if s > SIMILARITY_THRESHOLD)
                    }
        
        return quality_metrics
    
    def test_semantic_search(self, test_queries: List[str]) -> Dict:
        """Test the semantic search functionality with various queries."""
        search_results = {}
        
        for query in test_queries:
            analysis = self.chatbot._analyze_query(query)
            relevant_knowledge = self.chatbot._get_relevant_knowledge(analysis)
            search_results[query] = {
                'intent': analysis['intent'],
                'key_terms': analysis['key_terms'],
                'emotional_context': analysis['emotional_context'],
                'relevant_sections': list(relevant_knowledge.keys()) if isinstance(relevant_knowledge, dict) else []
            }
        
        return search_results
    
    def run_tests(self) -> Dict:
        """Run all knowledge base tests."""
        test_queries = [
            "Tell me about your artistic style",
            "What inspired your mythological paintings?",
            "How did you develop your technique?",
            "What materials did you prefer to work with?",
            "Who were your main influences?"
        ]
        
        # Run tests
        coverage = self.test_knowledge_coverage()
        quality = self.test_embedding_quality()
        search_results = self.test_semantic_search(test_queries)
        
        results = {
            'knowledge_coverage': coverage,
            'embedding_quality': quality,
            'semantic_search': search_results
        }
        
        print("\nKnowledge Coverage:", json.dumps(coverage, indent=2))
        print("\nEmbedding Quality:", json.dumps(quality, indent=2))
        print("\nSemantic Search Results:", json.dumps(search_results, indent=2))
        
        return results

if __name__ == "__main__":
    # Test with a specific artist
    artist_path = os.path.join(DATA_DIR, 'botticelli.json')
    tester = KnowledgeBaseTester(artist_path)
    results = tester.run_tests()
    
    # Save results to a JSON file
    results_file = os.path.join(REPORTS_DIR, 'knowledge_base_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)