import json
import random
import re
import nltk
import spacy
import os
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import local modules
from app.memory import ConversationMemory
from app.rewriter import rewrite_to_first_person

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

class ArtistChatbot:
    def __init__(self, json_path):
        """Initialize the chatbot with artist data from a JSON file."""
        # Load artist data
        with open(json_path, 'r') as f:
            self.artist_data = json.load(f)
        
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize sentence transformer for semantic search
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Extract artist information
        self.artist_name = self.artist_data.get('identity', {}).get('name', {}).get('full', 'Unknown Artist')
        self.personality = self.artist_data.get('identity', {}).get('personality_profile', {}).get('core_traits', [])
        
        # Initialize conversation memory
        self.memory = ConversationMemory(max_length=10)
        
        # Create knowledge base from JSON data
        self.knowledge_base = self._create_knowledge_base()
        
        # Create embeddings for knowledge base
        self.knowledge_embeddings = self._create_knowledge_embeddings()
        
        # Initialize conversation state
        self.conversation_focus = None
        self.previous_topics = []
        
        # Response quality thresholds
        self.min_confidence = 0.6
        self.max_attempts = 3
        
    def _create_knowledge_base(self) -> Dict:
        """Create a structured knowledge base from the JSON data."""
        knowledge = {
            'identity': {},
            'artistic': {},
            'philosophy': {},
            'technical': {},
            'personal': {},
            'masterworks': []
        }
        
        # Extract identity information
        identity = self.artist_data.get('identity', {})
        if identity:
            knowledge['identity'] = {
                'name': identity.get('name', {}),
                'birth': identity.get('vital_statistics', {}).get('birth', {}),
                'death': identity.get('vital_statistics', {}).get('death', {}),
                'background': identity.get('vital_statistics', {}).get('family_background', {}),
                'personality': identity.get('personality_profile', {}).get('core_traits', [])
            }
        
        # Extract artistic development
        artistic = self.artist_data.get('artistic_development', {})
        if artistic:
            knowledge['artistic'] = {
                'education': artistic.get('education', {}),
                'periods': artistic.get('artistic_periods', {}),
                'influences': artistic.get('influences', []),
                'methods': artistic.get('methods', [])
            }
        
        # Extract philosophical views
        philosophy = self.artist_data.get('artistic_philosophy', {})
        if philosophy:
            knowledge['philosophy'] = {
                'core_beliefs': philosophy.get('core_beliefs', {}),
                'views': philosophy.get('theoretical_framework', {}),
                'principles': philosophy.get('artistic_principles', [])
            }
        
        # Extract technical expertise
        technical = self.artist_data.get('technical_expertise', {})
        if technical:
            knowledge['technical'] = {
                'methods': technical.get('methods', []),
                'materials': technical.get('preferred_materials', []),
                'techniques': technical.get('techniques', {}),
                'innovations': technical.get('innovations', [])
            }
        
        # Extract personal life
        personal = self.artist_data.get('personal_life', {})
        if personal:
            knowledge['personal'] = {
                'daily_routine': personal.get('daily_routine', {}),
                'relationships': personal.get('relationships', {}),
                'interests': personal.get('interests', []),
                'beliefs': personal.get('beliefs', {})
            }
        
        # Extract masterworks
        masterworks = self.artist_data.get('masterworks', [])
        if masterworks:
            knowledge['masterworks'] = masterworks
        
        return knowledge
    
    def _create_knowledge_embeddings(self) -> Dict:
        """Create embeddings for the knowledge base."""
        embeddings = {}
        
        for category, data in self.knowledge_base.items():
            if isinstance(data, dict):
                embeddings[category] = {}
                for subcategory, content in data.items():
                    if isinstance(content, (str, list)):
                        text = content if isinstance(content, str) else ' '.join(content)
                        embeddings[category][subcategory] = self.model.encode([text])[0]
                    elif isinstance(content, dict):
                        embeddings[category][subcategory] = {}
                        for key, value in content.items():
                            if isinstance(value, (str, list)):
                                text = value if isinstance(value, str) else ' '.join(value)
                                embeddings[category][subcategory][key] = self.model.encode([text])[0]
            elif isinstance(data, list):
                embeddings[category] = []
                for item in data:
                    if isinstance(item, dict):
                        item_embedding = {}
                        for key, value in item.items():
                            if isinstance(value, (str, list)):
                                text = value if isinstance(value, str) else ' '.join(value)
                                item_embedding[key] = self.model.encode([text])[0]
                        embeddings[category].append(item_embedding)
        
        return embeddings
    
    def _analyze_query(self, query: str) -> Dict:
        """Analyze the user query to determine intent and extract key information."""
        doc = self.nlp(query.lower())
        
        # Extract entities and key phrases
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        
        # Determine query intent
        intent = self._determine_intent(doc)
        
        # Extract key terms
        key_terms = self._extract_key_terms(doc)
        
        # Extract emotional context
        emotional_context = self._extract_emotional_context(doc)
        
        return {
            'intent': intent,
            'entities': entities,
            'noun_chunks': noun_chunks,
            'key_terms': key_terms,
            'emotional_context': emotional_context,
            'original_query': query
        }
    
    def _determine_intent(self, doc) -> str:
        """Determine the primary intent of the query."""
        intent_keywords = {
            'identity': ['who', 'name', 'born', 'live', 'family', 'background'],
            'artistic': ['art', 'paint', 'work', 'style', 'technique', 'period'],
            'philosophy': ['think', 'believe', 'philosophy', 'view', 'opinion'],
            'technical': ['how', 'technique', 'method', 'material', 'process'],
            'historical': ['when', 'where', 'history', 'time', 'period', 'context'],
            'personal': ['life', 'daily', 'routine', 'relationship', 'interest', 'like', 'favorite']
        }
        
        # Score each intent based on keyword matches
        intent_scores = {intent: 0 for intent in intent_keywords}
        for token in doc:
            for intent, keywords in intent_keywords.items():
                if token.text in keywords:
                    intent_scores[intent] += 1
        
        # Return the highest scoring intent
        return max(intent_scores.items(), key=lambda x: x[1])[0]
    
    def _extract_key_terms(self, doc) -> List[str]:
        """Extract key terms from the query."""
        key_terms = []
        
        # Add noun chunks
        for chunk in doc.noun_chunks:
            key_terms.append(chunk.text)
        
        # Add named entities
        for ent in doc.ents:
            key_terms.append(ent.text)
        
        # Add important verbs
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "aux"]:
                key_terms.append(token.text)
        
        return list(set(key_terms))  # Remove duplicates
    
    def _extract_emotional_context(self, doc) -> str:
        """Extract emotional context from the query."""
        # Default emotional context
        emotional_context = "calmly"
        
        # Check for emotional indicators in the query
        emotional_indicators = {
            'excited': ['amazing', 'wonderful', 'incredible', 'fascinating', 'love'],
            'curious': ['curious', 'interested', 'wonder', 'tell me more'],
            'skeptical': ['doubt', 'skeptical', 'not sure', 'question'],
            'passionate': ['passionate', 'deeply', 'strongly', 'firmly'],
            'thoughtful': ['think', 'believe', 'consider', 'reflect']
        }
        
        query_text = doc.text.lower()
        for emotion, indicators in emotional_indicators.items():
            if any(indicator in query_text for indicator in indicators):
                emotional_context = emotion
                break
        
        return emotional_context
    
    def _get_relevant_knowledge(self, analysis: Dict) -> Dict:
        """Get relevant knowledge based on the query analysis."""
        intent = analysis['intent']
        key_terms = analysis['key_terms']
        
        # Get the relevant section of knowledge
        relevant_knowledge = self.knowledge_base.get(intent, {})
        
        # If the knowledge is empty, try to find relevant information in other sections
        if not relevant_knowledge:
            for section, data in self.knowledge_base.items():
                if section != intent and data:
                    # Check if any key terms match the data
                    for term in key_terms:
                        if self._term_in_data(term, data):
                            relevant_knowledge = data
                            break
                    if relevant_knowledge:
                        break
        
        return relevant_knowledge
    
    def _term_in_data(self, term: str, data: Any) -> bool:
        """Check if a term exists in the data structure."""
        if isinstance(data, str):
            return term.lower() in data.lower()
        elif isinstance(data, list):
            return any(self._term_in_data(term, item) for item in data)
        elif isinstance(data, dict):
            return any(self._term_in_data(term, value) for value in data.values())
        return False
    
    def _semantic_search(self, query: str, knowledge: Dict) -> List[str]:
        """Perform semantic search on the knowledge base."""
        query_embedding = self.model.encode([query])[0]
        
        # Flatten the knowledge structure for search
        flat_knowledge = self._flatten_knowledge(knowledge)
        
        # Calculate similarities
        similarities = []
        for text in flat_knowledge:
            text_embedding = self.model.encode([text])[0]
            similarity = cosine_similarity([query_embedding], [text_embedding])[0][0]
            similarities.append((text, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top 3 most relevant texts
        return [text for text, _ in similarities[:3] if _ > 0.5]
    
    def _flatten_knowledge(self, knowledge: Dict) -> List[str]:
        """Flatten a nested knowledge structure into a list of strings."""
        texts = []
        
        def extract_text(item):
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, list):
                for element in item:
                    extract_text(element)
            elif isinstance(item, dict):
                for value in item.values():
                    extract_text(value)
        
        extract_text(knowledge)
        return texts
    
    def _generate_response(self, analysis: Dict, relevant_knowledge: Dict) -> str:
        """Generate a response based on the query analysis and relevant knowledge."""
        intent = analysis['intent']
        emotional_context = analysis['emotional_context']
        key_terms = analysis['key_terms']
        
        # Get relevant texts through semantic search
        relevant_texts = self._semantic_search(analysis['original_query'], relevant_knowledge)
        
        # If no relevant texts found, use rule-based approach
        if not relevant_texts:
            return self._generate_rule_based_response(analysis, relevant_knowledge)
        
        # Extract key information from relevant texts
        key_info = self._extract_key_information(relevant_texts, key_terms)
        
        # Generate a dynamic response based on the extracted information
        response = self._construct_dynamic_response(key_info, intent, emotional_context)
        
        # Add personality trait in a natural way
        if self.personality:
            trait = random.choice(self.personality)
            response = self._integrate_personality(response, trait)
        
        return response
    
    def _extract_key_information(self, texts: List[str], key_terms: List[str]) -> Dict:
        """Extract key information from relevant texts based on key terms."""
        key_info = {
            'facts': [],
            'opinions': [],
            'techniques': [],
            'relationships': [],
            'events': []
        }
        
        for text in texts:
            # Extract facts (statements about the artist)
            if any(term in text.lower() for term in ['was', 'is', 'born', 'died', 'created', 'painted']):
                key_info['facts'].append(text)
            
            # Extract opinions (statements about beliefs or preferences)
            if any(term in text.lower() for term in ['believe', 'think', 'prefer', 'like', 'enjoy', 'favorite']):
                key_info['opinions'].append(text)
            
            # Extract techniques (statements about artistic methods)
            if any(term in text.lower() for term in ['technique', 'method', 'style', 'approach', 'process']):
                key_info['techniques'].append(text)
            
            # Extract relationships (statements about connections with others)
            if any(term in text.lower() for term in ['relationship', 'friend', 'mentor', 'student', 'influence']):
                key_info['relationships'].append(text)
            
            # Extract events (statements about specific occurrences)
            if any(term in text.lower() for term in ['event', 'exhibition', 'commission', 'travel', 'visit']):
                key_info['events'].append(text)
        
        # Prioritize information that matches key terms
        for category in key_info:
            key_info[category] = sorted(
                key_info[category],
                key=lambda x: sum(1 for term in key_terms if term in x.lower()),
                reverse=True
            )
        
        return key_info
    
    def _construct_dynamic_response(self, key_info: Dict, intent: str, emotional_context: str) -> str:
        """Construct a dynamic response based on extracted information."""
        response_parts = []
        
        # Select information based on intent
        if intent == 'identity':
            if key_info['facts']:
                response_parts.append(key_info['facts'][0])
            if key_info['events']:
                response_parts.append(key_info['events'][0])
        
        elif intent == 'artistic':
            if key_info['techniques']:
                response_parts.append(key_info['techniques'][0])
            if key_info['opinions']:
                response_parts.append(key_info['opinions'][0])
        
        elif intent == 'philosophy':
            if key_info['opinions']:
                response_parts.append(key_info['opinions'][0])
            if key_info['facts'] and len(key_info['facts']) > 1:
                response_parts.append(key_info['facts'][1])
        
        elif intent == 'technical':
            if key_info['techniques']:
                response_parts.append(key_info['techniques'][0])
            if key_info['facts'] and any('material' in f.lower() or 'technique' in f.lower() for f in key_info['facts']):
                for fact in key_info['facts']:
                    if 'material' in fact.lower() or 'technique' in fact.lower():
                        response_parts.append(fact)
                        break
        
        elif intent == 'personal':
            if key_info['relationships']:
                response_parts.append(key_info['relationships'][0])
            if key_info['opinions']:
                response_parts.append(key_info['opinions'][0])
        
        else:
            # Default: combine different types of information
            for category in ['facts', 'opinions', 'techniques', 'relationships', 'events']:
                if key_info[category]:
                    response_parts.append(key_info[category][0])
                    break
        
        # If no specific information found, use a fallback
        if not response_parts:
            return self._get_fallback_response(intent)
        
        # Combine response parts
        response = ". ".join(response_parts)
        
        # Add emotional context if not calm
        if emotional_context != "calmly":
            response = f"{emotional_context}, {response.lower()}"
        
        return response
    
    def _integrate_personality(self, response: str, trait: str) -> str:
        """Integrate personality trait into the response in a natural way."""
        # Different ways to integrate personality
        integration_methods = [
            lambda r, t: f"{r} This reflects my {t.lower()} nature.",
            lambda r, t: f"{r} My {t.lower()} approach is evident in this.",
            lambda r, t: f"{r} As a {t.lower()} artist, I approach this differently than others might.",
            lambda r, t: f"{r} This demonstrates my {t.lower()} perspective on art and life."
        ]
        
        return random.choice(integration_methods)(response, trait)
    
    def _get_fallback_response(self, intent: str) -> str:
        """Get a fallback response when no specific information is available."""
        fallbacks = {
            'identity': "I have a unique identity that has evolved throughout my life and career.",
            'artistic': "My artistic style is distinctive and has developed over many years of practice.",
            'philosophy': "My philosophical views on art and life are deeply personal and have shaped my work.",
            'technical': "I have developed my own technical approaches through years of experimentation.",
            'personal': "My personal life has been rich with experiences that have influenced my art.",
            'default': "I have many interesting experiences and perspectives to share about my art and life."
        }
        
        return fallbacks.get(intent, fallbacks['default'])
    
    def _generate_rule_based_response(self, analysis: Dict, relevant_knowledge: Dict) -> str:
        """Generate a response using rule-based approach when semantic search fails."""
        intent = analysis['intent']
        emotional_context = analysis['emotional_context']
        
        # Get a personality trait
        trait = random.choice(self.personality) if self.personality else "I"
        
        # Generate response based on intent
        if intent == 'identity':
            return self._generate_identity_response(relevant_knowledge, trait, emotional_context)
        elif intent == 'artistic':
            return self._generate_artistic_response(relevant_knowledge, trait, emotional_context)
        elif intent == 'philosophy':
            return self._generate_philosophy_response(relevant_knowledge, trait, emotional_context)
        elif intent == 'technical':
            return self._generate_technical_response(relevant_knowledge, trait, emotional_context)
        elif intent == 'personal':
            return self._generate_personal_response(relevant_knowledge, trait, emotional_context)
        else:
            return self._get_random_fact(relevant_knowledge)
    
    def _generate_identity_response(self, knowledge: Dict, trait: str, emotional_context: str) -> str:
        """Generate a response for identity-related questions."""
        name = knowledge.get('name', {}).get('full', self.artist_name)
        birth = knowledge.get('birth', {})
        birth_date = birth.get('date', '')
        birth_place = birth.get('place', '')
        
        # Create a more natural response
        response_parts = []
        
        if birth_date and birth_place:
            response_parts.append(f"I was born on {birth_date} in {birth_place}")
        elif birth_date:
            response_parts.append(f"I was born on {birth_date}")
        elif birth_place:
            response_parts.append(f"I was born in {birth_place}")
        
        if name and name != "Unknown Artist":
            response_parts.append(f"My name is {name}")
        
        # Add emotional context and personality trait naturally
        if response_parts:
            response = ". ".join(response_parts) + "."
            if emotional_context != "calmly":
                response = f"{emotional_context}, {response.lower()}"
            return response
        else:
            return f"I am {name}."
    
    def _generate_artistic_response(self, knowledge: Dict, trait: str, emotional_context: str) -> str:
        """Generate a response for artistic questions."""
        # Check if the query is about specific paintings
        if any(term in self.current_query.lower() for term in ['paint', 'painting', 'artwork', 'work', 'piece']):
            return self._generate_painting_response(knowledge, trait, emotional_context)
        
        methods = knowledge.get('methods', [])
        influences = knowledge.get('influences', [])
        
        # Create a more natural response
        response_parts = []
        
        if methods:
            response_parts.append(f"My artistic approach involves {methods[0]}")
        
        if influences:
            response_parts.append(f"I was influenced by {influences[0]}")
        
        # Add emotional context and personality trait naturally
        if response_parts:
            response = ". ".join(response_parts) + "."
            if emotional_context != "calmly":
                response = f"{emotional_context}, {response.lower()}"
            return response
        else:
            return f"I have my own unique artistic style."
    
    def _generate_painting_response(self, knowledge: Dict, trait: str, emotional_context: str) -> str:
        """Generate a response specifically about paintings."""
        masterworks = knowledge.get('masterworks', {})
        if not masterworks:
            return "I created many paintings throughout my life, each expressing different aspects of my experiences and emotions."
        
        # Select a random painting to discuss
        painting_key = random.choice(list(masterworks.keys()))
        painting = masterworks[painting_key]
        
        # Format the painting title
        title = painting_key.replace('_', ' ').title()
        
        # Build response about the painting
        response_parts = []
        
        # Add date and location
        if 'date' in painting:
            response_parts.append(f"I painted {title} in {painting['date']}")
        
        # Add significance
        if 'significance' in painting and painting['significance']:
            response_parts.append(painting['significance'][0])
        
        # Add technical aspects
        if 'technical_aspects' in painting and painting['technical_aspects']:
            response_parts.append(f"I used {painting['technical_aspects'][0].lower()}")
        
        # Combine response parts
        if response_parts:
            response = ". ".join(response_parts) + "."
            if emotional_context != "calmly":
                response = f"{emotional_context}, {response.lower()}"
            return response
        
        return f"I painted {title}, which is one of my significant works."
    
    def _generate_philosophy_response(self, knowledge: Dict, trait: str, emotional_context: str) -> str:
        """Generate a response for philosophical questions."""
        beliefs = knowledge.get('core_beliefs', {})
        principles = knowledge.get('principles', [])
        
        # Create a more natural response
        response_parts = []
        
        if beliefs and isinstance(beliefs, dict):
            for key, value in beliefs.items():
                if value:
                    response_parts.append(f"I believe that {value}")
        
        if principles:
            response_parts.append(f"I follow the principle that {principles[0]}")
        
        # Add emotional context and personality trait naturally
        if response_parts:
            response = ". ".join(response_parts) + "."
            if emotional_context != "calmly":
                response = f"{emotional_context}, {response.lower()}"
            return response
        else:
            return f"I have my own philosophical views on art and life."
    
    def _generate_technical_response(self, knowledge: Dict, trait: str, emotional_context: str) -> str:
        """Generate a response for technical questions."""
        methods = knowledge.get('methods', [])
        materials = knowledge.get('materials', [])
        
        # Create a more natural response
        response_parts = []
        
        if methods and materials:
            response_parts.append(f"I prefer working with {materials[0]} and using {methods[0]}")
        elif methods:
            response_parts.append(f"I prefer using {methods[0]} in my work")
        elif materials:
            response_parts.append(f"I prefer working with {materials[0]}")
        
        # Add emotional context and personality trait naturally
        if response_parts:
            response = ". ".join(response_parts) + "."
            if emotional_context != "calmly":
                response = f"{emotional_context}, {response.lower()}"
            return response
        else:
            return f"I have developed my own technical approach."
    
    def _generate_personal_response(self, knowledge: Dict, trait: str, emotional_context: str) -> str:
        """Generate a response for personal questions."""
        interests = knowledge.get('interests', [])
        relationships = knowledge.get('relationships', {})
        
        # Create a more natural response
        response_parts = []
        
        if interests:
            response_parts.append(f"I enjoy {interests[0]}")
        
        if relationships:
            for key, value in relationships.items():
                if value:
                    response_parts.append(f"My relationship with {key} was {value}")
        
        # Add emotional context and personality trait naturally
        if response_parts:
            response = ". ".join(response_parts) + "."
            if emotional_context != "calmly":
                response = f"{emotional_context}, {response.lower()}"
            return response
        else:
            return f"I have many personal interests and relationships."
    
    def _get_random_fact(self, knowledge: Dict) -> str:
        """Get a random fact from the knowledge base."""
        facts = []
        
        for key, value in knowledge.items():
            if isinstance(value, list) and value:
                facts.append(f"My {key} include {value[0]}")
            elif isinstance(value, dict):
                for k, v in value.items():
                    if v:
                        facts.append(f"My {k} is {v}")
            elif value:
                facts.append(f"My {key} is {value}")
        
        if facts:
            return random.choice(facts)
        else:
            return "I have many interesting experiences to share"
    
    def _calculate_response_confidence(self, response: str, query: str, relevant_knowledge: Dict) -> float:
        """Calculate confidence score for a response."""
        # Calculate semantic similarity between query and response
        query_embedding = self.model.encode([query])[0]
        response_embedding = self.model.encode([response])[0]
        semantic_similarity = float(np.dot(query_embedding, response_embedding) / 
                                 (np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding)))
        
        # Calculate knowledge coverage
        knowledge_coverage = len(relevant_knowledge) / len(self.knowledge_base)
        
        # Calculate personality consistency
        personality_embedding = self.model.encode([' '.join(self.personality)])[0]
        personality_consistency = float(np.dot(response_embedding, personality_embedding) / 
                                     (np.linalg.norm(response_embedding) * np.linalg.norm(personality_embedding)))
        
        # Calculate response length score (prefer medium-length responses)
        words = response.split()
        length_score = 1.0 - abs(len(words) - 100) / 100  # Optimal length around 100 words
        
        # Combine scores with weights
        confidence = (
            0.4 * semantic_similarity +
            0.3 * knowledge_coverage +
            0.2 * personality_consistency +
            0.1 * length_score
        )
        
        return float(confidence)
    
    def _improve_response(self, response: str, query: str, analysis: Dict, relevant_knowledge: Dict) -> Tuple[str, float]:
        """Attempt to improve response quality through multiple iterations."""
        best_response = response
        best_confidence = self._calculate_response_confidence(response, query, relevant_knowledge)
        
        for attempt in range(self.max_attempts):
            # Generate alternative response
            alt_response = self._generate_response(analysis, relevant_knowledge)
            confidence = self._calculate_response_confidence(alt_response, query, relevant_knowledge)
            
            # Update best response if confidence improved
            if confidence > best_confidence:
                best_response = alt_response
                best_confidence = confidence
            
            # Early exit if confidence is high enough
            if best_confidence >= self.min_confidence:
                break
        
        return best_response, best_confidence
    
    def respond(self, query: str) -> str:
        """Generate a response to the user query."""
        # Analyze query
        analysis = self._analyze_query(query)
        
        # Get relevant knowledge
        relevant_knowledge = self._get_relevant_knowledge(analysis)
        
        # Generate initial response
        initial_response = self._generate_response(analysis, relevant_knowledge)
        
        # Improve response quality
        final_response, confidence = self._improve_response(
            initial_response, query, analysis, relevant_knowledge
        )
        
        # Add to conversation memory
        self.memory.add_turn(
            user_input=query,
            bot_response=final_response,
            confidence=confidence,
            intent=analysis['intent'],
            emotional_context=analysis['emotional_context']
        )
        
        return final_response

# Example usage
if __name__ == "__main__":
    chatbot = ArtistChatbot("data/kahlo.json")
    
    print("Chat initialized. Type 'exit' to end the conversation.")
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            break
        response = chatbot.respond(query)
        print(f"Artist: {response}") 