import os
import json
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from test_config import (
    DATA_DIR, REPORTS_DIR, EMBEDDING_MODEL,
    MIN_RESPONSE_LENGTH, MAX_RESPONSE_LENGTH
)
from src.chatbot import ArtistChatbot

class ResponseQualityTester:
    def __init__(self, artist_data_path: str):
        self.chatbot = ArtistChatbot(artist_data_path)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
    def test_response_relevance(self, query: str, response: str) -> float:
        """Test how relevant the response is to the query using semantic similarity."""
        query_embedding = self.model.encode([query])[0]
        response_embedding = self.model.encode([response])[0]
        similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]
        return float(similarity)
    
    def test_response_length(self, response: str) -> Dict:
        """Test the length characteristics of the response."""
        words = response.split()
        sentences = response.split('.')
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'is_within_length_limits': MIN_RESPONSE_LENGTH <= len(words) <= MAX_RESPONSE_LENGTH
        }
    
    def test_personality_consistency(self, response: str) -> float:
        """Test how well the response maintains the artist's personality."""
        personality_traits = self.chatbot.personality
        trait_embeddings = self.model.encode(personality_traits)
        response_embedding = self.model.encode([response])[0]
        
        similarities = [cosine_similarity([response_embedding], [trait_embedding])[0][0] 
                       for trait_embedding in trait_embeddings]
        return float(np.max(similarities))
    
    def evaluate_response(self, query: str) -> Dict:
        """Evaluate a response using multiple metrics."""
        response = self.chatbot.respond(query)
        
        return {
            'query': query,
            'response': response,
            'relevance_score': self.test_response_relevance(query, response),
            'length_metrics': self.test_response_length(response),
            'personality_consistency': self.test_personality_consistency(response)
        }
    
    def run_tests(self) -> List[Dict]:
        """Run all response quality tests."""
        test_queries = [
            "What inspired your artistic style?",
            "How did you develop your technique?",
            "What materials did you prefer to work with?",
            "Who were your main influences?",
            "What was your creative process like?"
        ]
        
        results = []
        for query in test_queries:
            result = self.evaluate_response(query)
            results.append(result)
            
            print(f"\nTesting query: {query}")
            print(f"Response: {result['response']}")
            print(f"Relevance Score: {result['relevance_score']:.3f}")
            print(f"Personality Consistency: {result['personality_consistency']:.3f}")
            print("Length Metrics:", result['length_metrics'])
        
        return results

if __name__ == "__main__":
    # Test with a specific artist
    artist_path = os.path.join(DATA_DIR, 'botticelli.json')
    tester = ResponseQualityTester(artist_path)
    results = tester.run_tests()
    
    # Save results to a JSON file
    results_file = os.path.join(REPORTS_DIR, 'response_quality_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)