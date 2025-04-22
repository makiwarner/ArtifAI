import os
import json
import spacy
import nltk
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# List of supported artists
SUPPORTED_ARTISTS = [
    'botticelli',
    'da_vinci',
    'kahlo',
    'klimt',
    'monet',
    'picasso',
    'rivera',
    'van_gogh',
    'vermeer'
]

class Retriever:
    def __init__(self):
        self.artists_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        self.artists_data = {}
        self.conversation_history = {}  # Initialize conversation history first
        self.nlp = spacy.load("en_core_web_md")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))
        self._load_artists()

    def _load_artists(self):
        """Load all artist data from JSON files in the data directory."""
        for artist_id in SUPPORTED_ARTISTS:
            file_path = os.path.join(self.artists_dir, f"{artist_id}.json")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.artists_data[artist_id] = json.load(f)
                    # Initialize conversation history for this artist
                    self.conversation_history[artist_id] = []
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not load {artist_id}.json - {str(e)}")
                    continue

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        doc = self.nlp(text)
        
        # Remove stopwords and lemmatize
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities and key phrases from text."""
        doc = self.nlp(text)
        entities = {
            'artwork': [],
            'technique': [],
            'person': [],
            'date': [],
            'location': [],
            'misc': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON']:
                entities['person'].append(ent.text)
            elif ent.label_ in ['DATE', 'TIME']:
                entities['date'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['location'].append(ent.text)
            else:
                entities['misc'].append(ent.text)
        
        # Extract potential artwork names (capitalized phrases)
        for chunk in doc.noun_chunks:
            if chunk.text[0].isupper():
                entities['artwork'].append(chunk.text)
        
        return entities

    def get_relevant_knowledge(self, artist_data: Dict, query: str, entities: Dict[str, List[str]]) -> List[str]:
        """Get relevant knowledge based on the query and extracted entities."""
        knowledge = []
        
        # Add relevant sections based on entity types
        if entities['artwork']:
            if 'artworks' in artist_data:
                knowledge.extend(self._extract_text(artist_data['artworks']))
        
        if entities['technique']:
            if 'techniqueAndMaterials' in artist_data:
                knowledge.extend(self._extract_text(artist_data['techniqueAndMaterials']))
        
        # Always include core knowledge
        for key in ['knowledge', 'bio', 'FAQ']:
            if key in artist_data:
                knowledge.extend(self._extract_text(artist_data[key]))
        
        return knowledge

    def _extract_text(self, content) -> List[str]:
        """Recursively extract text from nested structures."""
        texts = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                texts.extend(self._extract_text(item))
        elif isinstance(content, dict):
            for value in content.values():
                texts.extend(self._extract_text(value))
        return texts

    ARTIST_DISPLAY_NAMES = {
        'botticelli': 'Sandro Botticelli',
        'da_vinci': 'Leonardo da Vinci',
        'kahlo': 'Frida Kahlo',
        'klimt': 'Gustav Klimt',
        'monet': 'Claude Monet',
        'picasso': 'Pablo Picasso',
        'rivera': 'Diego Rivera',
        'van_gogh': 'Vincent van Gogh',
        'vermeer': 'Johannes Vermeer'
    }

    def get_artist_name(self, artist_id: str) -> str:
        """Get the name of an artist by their ID."""
        return self.ARTIST_DISPLAY_NAMES.get(artist_id, 'Unknown Artist')

    def get_available_artists(self) -> List[Dict]:
        """Return a list of available artists with their basic information."""
        artists = []
        for artist_id, data in self.artists_data.items():
            name = self.ARTIST_DISPLAY_NAMES.get(artist_id, 'Unknown Artist')
            artists.append({
                'id': artist_id,
                'name': name,
                'bio': data.get('bio', ''),
                'image_url': f'/static/images/{artist_id}.jpg'
            })
        return sorted(artists, key=lambda x: x['name'])

    def get_response(self, artist_id: str, message: str) -> str:
        """Get a response from the artist based on the user's message."""
        # Get artist data
        artist_data = self.artists_data.get(artist_id)
        if not artist_data:
            return "I'm sorry, I couldn't find information about this artist."

        # Preprocess the user's message
        processed_message = self.preprocess_text(message)
        
        # Extract entities and key information
        entities = self.extract_entities(message)
        
        # Get relevant knowledge based on the query and entities
        knowledge = self.get_relevant_knowledge(artist_data, processed_message, entities)
        
        if not knowledge:
            return "I'm not sure how to respond to that. Could you try asking something else about my art or life?"

        # Convert knowledge to embeddings
        knowledge_embeddings = self.model.encode(knowledge)
        message_embedding = self.model.encode([processed_message])[0]

        # Calculate similarities
        similarities = cosine_similarity([message_embedding], knowledge_embeddings)[0]
        
        # Get the top 3 most relevant responses
        top_indices = np.argsort(similarities)[-3:][::-1]
        best_responses = [knowledge[i] for i in top_indices if similarities[i] > 0.6]
        
        if not best_responses:
            return "I'm not sure how to respond to that. Could you try asking something else about my art or life?"
            
        # Update conversation history
        self.conversation_history[artist_id].append({
            'user': message,
            'response': best_responses[0]
        })
        
        # Return the best response
        return best_responses[0]

    def get_conversation_history(self, artist_id: str) -> List[Dict]:
        """Get the conversation history for a specific artist."""
        return self.conversation_history.get(artist_id, [])