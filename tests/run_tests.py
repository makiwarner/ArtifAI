import os
import json
import time
import sys
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from test_config import (
    REPORTS_DIR, TEST_ARTISTS, DATA_DIR,
    MIN_RESPONSE_LENGTH, MAX_RESPONSE_LENGTH,
    SIMILARITY_THRESHOLD, MAX_MEMORY_TURNS
)
from test_response_quality import ResponseQualityTester
from test_knowledge_base import KnowledgeBaseTester
from test_conversation_memory import ConversationMemoryTester

def run_all_tests() -> Dict:
    """Run all test suites and collect results."""
    start_time = time.time()
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_parameters': {
            'min_response_length': MIN_RESPONSE_LENGTH,
            'max_response_length': MAX_RESPONSE_LENGTH,
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'max_memory_turns': MAX_MEMORY_TURNS
        },
        'artists_tested': TEST_ARTISTS,
        'results': {}
    }
    
    for artist_file in TEST_ARTISTS:
        artist_path = os.path.join(DATA_DIR, artist_file)
        artist_name = os.path.splitext(artist_file)[0]
        print(f"\nTesting {artist_name}...")
        
        # Run response quality tests
        print("Running response quality tests...")
        response_tester = ResponseQualityTester(artist_path)
        response_results = response_tester.run_tests()
        
        # Run knowledge base tests
        print("Running knowledge base tests...")
        knowledge_tester = KnowledgeBaseTester(artist_path)
        knowledge_results = knowledge_tester.run_tests()
        
        # Run conversation memory tests
        print("Running conversation memory tests...")
        memory_tester = ConversationMemoryTester(artist_path)
        memory_results = memory_tester.run_tests()
        
        results['results'][artist_name] = {
            'response_quality': response_results,
            'knowledge_base': knowledge_results,
            'conversation_memory': memory_results
        }
    
    results['total_time'] = time.time() - start_time
    return results

def generate_summary_report(results: Dict[str, Dict]) -> None:
    """Generate a summary report of all test results."""
    print("\nGenerating summary report...")
    
    # Calculate average response quality metrics
    response_metrics = results.get('response_quality', {})
    avg_relevance = sum(response_metrics.get('relevance_scores', [])) / len(response_metrics.get('relevance_scores', [1]))
    avg_personality = sum(response_metrics.get('personality_scores', [])) / len(response_metrics.get('personality_scores', [1]))
    
    # Calculate knowledge base metrics
    knowledge_metrics = results.get('knowledge_base', {})
    coverage_data = knowledge_metrics.get('coverage', {})
    embedding_data = knowledge_metrics.get('embedding_quality', {})
    
    # Calculate average coverage across all sections
    total_fields = sum(section.get('fields', 0) for section in coverage_data.values())
    total_non_empty = sum(section.get('non_empty_fields', 0) for section in coverage_data.values())
    avg_coverage = total_non_empty / total_fields if total_fields > 0 else 0
    
    # Calculate average embedding quality
    avg_embedding_quality = sum(
        section.get('avg_similarity', 0) 
        for section in embedding_data.values()
    ) / len(embedding_data) if embedding_data else 0
    
    # Calculate conversation memory metrics
    memory_metrics = results.get('conversation_memory', {})
    context_retention = memory_metrics.get('context_retention', {})
    memory_overflow = memory_metrics.get('memory_overflow', {})
    context_switching = memory_metrics.get('context_switching', {})
    
    # Calculate average context retention
    avg_topic_consistency = sum(context_retention.get('topic_consistency', [0])) / len(context_retention.get('topic_consistency', [1]))
    avg_reference_resolution = sum(context_retention.get('reference_resolution', [0])) / len(context_retention.get('reference_resolution', [1]))
    
    # Calculate memory overflow impact
    memory_sizes = memory_overflow.get('memory_size', [])
    response_qualities = memory_overflow.get('response_quality', [])
    memory_impact = sum(response_qualities) / len(response_qualities) if response_qualities else 0
    
    # Calculate context switching effectiveness
    topic_transitions = context_switching.get('topic_transitions', [])
    response_relevance = context_switching.get('response_relevance', [])
    avg_switch_quality = sum(response_relevance) / len(response_relevance) if response_relevance else 0
    
    # Create summary report
    summary = {
        'response_quality': {
            'average_relevance': avg_relevance,
            'average_personality_consistency': avg_personality,
            'overall_score': (avg_relevance + avg_personality) / 2
        },
        'knowledge_base': {
            'average_coverage': avg_coverage,
            'average_embedding_quality': avg_embedding_quality,
            'overall_score': (avg_coverage + avg_embedding_quality) / 2
        },
        'conversation_memory': {
            'context_retention': {
                'topic_consistency': avg_topic_consistency,
                'reference_resolution': avg_reference_resolution,
                'overall_score': (avg_topic_consistency + avg_reference_resolution) / 2
            },
            'memory_overflow': {
                'average_impact': memory_impact,
                'max_memory_size': max(memory_sizes) if memory_sizes else 0
            },
            'context_switching': {
                'average_quality': avg_switch_quality,
                'transition_count': len(topic_transitions)
            },
            'overall_score': (avg_topic_consistency + avg_reference_resolution + memory_impact + avg_switch_quality) / 4
        }
    }
    
    # Calculate overall system score
    overall_score = (
        summary['response_quality']['overall_score'] +
        summary['knowledge_base']['overall_score'] +
        summary['conversation_memory']['overall_score']
    ) / 3
    
    summary['overall_system_score'] = overall_score
    
    # Save summary report
    with open(os.path.join(REPORTS_DIR, 'summary_report.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nTest Summary:")
    print(f"Overall System Score: {overall_score:.2f}")
    print("\nResponse Quality:")
    print(f"- Average Relevance: {avg_relevance:.2f}")
    print(f"- Average Personality Consistency: {avg_personality:.2f}")
    print("\nKnowledge Base:")
    print(f"- Average Coverage: {avg_coverage:.2f}")
    print(f"- Average Embedding Quality: {avg_embedding_quality:.2f}")
    print("\nConversation Memory:")
    print(f"- Context Retention Score: {summary['conversation_memory']['context_retention']['overall_score']:.2f}")
    print(f"- Memory Overflow Impact: {memory_impact:.2f}")
    print(f"- Context Switching Quality: {avg_switch_quality:.2f}")
    
    print(f"\nDetailed results saved to {os.path.join(REPORTS_DIR, 'summary_report.json')}")

def main():
    """Main function to run all tests and generate reports."""
    print("Starting test suite...")
    results = run_all_tests()
    
    print("\nGenerating summary report...")
    generate_summary_report(results)
    
    print(f"\nTests completed in {results['total_time']:.2f} seconds")
    print(f"Results saved in {REPORTS_DIR}")
    print(f"Summary report available in {os.path.join(REPORTS_DIR, 'summary')}")

if __name__ == "__main__":
    main() 