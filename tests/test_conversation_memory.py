import os
import json
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from test_config import (
    DATA_DIR, REPORTS_DIR, EMBEDDING_MODEL,
    MAX_MEMORY_TURNS
)
from src.chatbot import ArtistChatbot
from src.app.memory import ConversationMemory

class ConversationMemoryTester:
    def __init__(self, artist_data_path: str):
        self.chatbot = ArtistChatbot(artist_data_path)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
    def test_context_retention(self, conversation: List[str]) -> Dict:
        """Test how well the chatbot maintains context across a conversation."""
        context_metrics = {
            'topic_consistency': [],
            'reference_resolution': [],
            'memory_usage': []
        }
        
        for i, query in enumerate(conversation):
            response = self.chatbot.respond(query)
            
            # Test topic consistency
            if i > 0:
                # Get the last turn from memory
                last_turn = self.chatbot.memory.get_context()[-1]
                prev_response = last_turn.get('bot_response', '')  # Use get() with default value
                topic_similarity = self._calculate_similarity(prev_response, response)
                context_metrics['topic_consistency'].append(float(topic_similarity))  # Convert to Python float
            
            # Test reference resolution
            if i > 0:
                prev_query = conversation[i-1]
                reference_resolution = self._test_reference_resolution(prev_query, query, response)
                context_metrics['reference_resolution'].append(float(reference_resolution))  # Convert to Python float
            
            # Track memory usage
            context_metrics['memory_usage'].append(len(self.chatbot.memory.get_context()))
        
        return context_metrics
    
    def test_memory_overflow(self, max_turns: int = MAX_MEMORY_TURNS) -> Dict:
        """Test how the chatbot handles memory overflow."""
        overflow_metrics = {
            'memory_size': [],
            'response_quality': []
        }
        
        # Generate a long conversation
        base_query = "Tell me about your artistic style"
        for i in range(max_turns):
            query = f"{base_query} - turn {i+1}"
            response = self.chatbot.respond(query)
            
            # Track memory size
            overflow_metrics['memory_size'].append(len(self.chatbot.memory.get_context()))
            
            # Test response quality
            response_quality = self._test_response_quality(query, response)
            overflow_metrics['response_quality'].append(float(response_quality))  # Convert to Python float
        
        return overflow_metrics
    
    def test_context_switching(self, conversations: List[List[str]]) -> Dict:
        """Test how well the chatbot handles switching between different topics."""
        switching_metrics = {
            'topic_transitions': [],
            'response_relevance': []
        }
        
        for conversation in conversations:
            for i, query in enumerate(conversation):
                response = self.chatbot.respond(query)
                
                if i > 0:
                    # Test topic transition
                    prev_query = conversation[i-1]
                    topic_transition = self._calculate_similarity(prev_query, query)
                    switching_metrics['topic_transitions'].append(float(topic_transition))  # Convert to Python float
                    
                    # Test response relevance
                    relevance = self._calculate_similarity(query, response)
                    switching_metrics['response_relevance'].append(float(relevance))  # Convert to Python float
        
        return switching_metrics
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        embedding1 = self.model.encode([text1])[0]
        embedding2 = self.model.encode([text2])[0]
        return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    
    def _test_reference_resolution(self, prev_query: str, current_query: str, response: str) -> float:
        """Test how well the chatbot resolves references to previous context."""
        prev_embedding = self.model.encode([prev_query])[0]
        response_embedding = self.model.encode([response])[0]
        return float(np.dot(prev_embedding, response_embedding) / (np.linalg.norm(prev_embedding) * np.linalg.norm(response_embedding)))
    
    def _test_response_quality(self, query: str, response: str) -> float:
        """Test the quality of the response in relation to the query."""
        return self._calculate_similarity(query, response)
    
    def run_tests(self) -> Dict:
        """Run all conversation memory tests."""
        # Test conversations
        conversations = [
            # Conversation 1: Artistic style discussion
            [
                "What is your artistic style?",
                "How did you develop this style?",
                "What materials did you use to achieve this effect?",
                "Who influenced your technique?"
            ],
            # Conversation 2: Memory overflow test
            ["Tell me about your work"] * MAX_MEMORY_TURNS,
            # Conversation 3: Context switching
            [
                "What inspired your mythological paintings?",
                "How did you choose your subjects?",
                "What was your relationship with the Medici family?",
                "How did you handle the commission for the Sistine Chapel?"
            ]
        ]
        
        # Run tests
        context_retention = self.test_context_retention(conversations[0])
        memory_overflow = self.test_memory_overflow()
        context_switching = self.test_context_switching(conversations[2])
        
        results = {
            'context_retention': context_retention,
            'memory_overflow': memory_overflow,
            'context_switching': context_switching
        }
        
        print("\nContext Retention:", json.dumps(context_retention, indent=2))
        print("\nMemory Overflow:", json.dumps(memory_overflow, indent=2))
        print("\nContext Switching:", json.dumps(context_switching, indent=2))
        
        return results

if __name__ == "__main__":
    # Test with a specific artist
    artist_path = os.path.join(DATA_DIR, 'botticelli.json')
    tester = ConversationMemoryTester(artist_path)
    results = tester.run_tests()
    
    # Save results to a JSON file
    results_file = os.path.join(REPORTS_DIR, 'conversation_memory_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)