import json
import random
import re
import nltk
import spacy
from typing import Dict, List, Any, Tuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

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
        with open(json_path, 'r') as f:
            self.artist_data = json.load(f)
        
        # Get artist name from the new structure
        self.artist_name = self.artist_data.get('identity', {}).get('name', {}).get('full', 'Unknown Artist')
        self.personality = self.artist_data.get('identity', {}).get('personality_profile', {}).get('core_traits', [])
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize conversation state
        self.conversation_focus = None
        self.previous_topics = []
        
        # Define topic keywords
        self.topic_keywords = {
            'identity': ['who', 'name', 'born', 'live', 'family', 'background'],
            'artistic': ['art', 'paint', 'work', 'style', 'technique', 'period'],
            'philosophy': ['think', 'believe', 'philosophy', 'view', 'opinion'],
            'technical': ['how', 'technique', 'method', 'material', 'process'],
            'historical': ['when', 'where', 'history', 'time', 'period', 'context'],
            'personal': ['life', 'daily', 'routine', 'relationship', 'interest']
        }

    def _preprocess_text(self, text):
        """Preprocess the input text using NLP techniques."""
        # Tokenize and lemmatize
        tokens = word_tokenize(text.lower())
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove stop words
        filtered = [word for word in lemmatized if word not in self.stop_words]
        
        # POS tagging
        tagged = pos_tag(filtered)
        
        return filtered, tagged

    def _analyze_query(self, query):
        """Analyze the user query to determine the topic and extract key information."""
        filtered, tagged = self._preprocess_text(query)
        
        # Determine topic based on keywords
        topic_scores = {topic: 0 for topic in self.topic_keywords}
        for word in filtered:
            for topic, keywords in self.topic_keywords.items():
                if word in keywords:
                    topic_scores[topic] += 1
        
        # Get the most relevant topic
        main_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
        
        # Extract named entities using spaCy
        doc = self.nlp(query)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return main_topic, entities

    def _get_relevant_content(self, topic, entities):
        """Retrieve relevant content from the artist data based on topic and entities."""
        if topic == 'identity':
            return self._extract_identity_info(entities)
        elif topic == 'artistic':
            return self._extract_artistic_info(entities)
        elif topic == 'philosophy':
            return self._extract_philosophy_info(entities)
        elif topic == 'technical':
            return self._extract_technical_info(entities)
        elif topic == 'historical':
            return self._extract_historical_info(entities)
        elif topic == 'personal':
            return self._extract_personal_info(entities)
        return None

    def _extract_identity_info(self, entities):
        """Extract relevant identity information."""
        identity = self.artist_data.get('identity', {})
        vital_stats = identity.get('vital_statistics', {})
        return {
            'name': identity.get('name', {}).get('full', 'Unknown'),
            'birth': vital_stats.get('birth', {}).get('date', 'unknown'),
            'death': vital_stats.get('death', {}).get('date', ''),
            'background': vital_stats.get('family_background', {}),
            'traits': identity.get('personality_profile', {}).get('core_traits', [])
        }

    def _extract_artistic_info(self, entities):
        """Extract relevant artistic information."""
        artistic = self.artist_data.get('artistic_development', {})
        return {
            'education': artistic.get('education', {}),
            'periods': artistic.get('artistic_periods', {}),
            'works': self.artist_data.get('masterworks', [])
        }

    def _extract_philosophy_info(self, entities):
        """Extract relevant philosophical information."""
        return self.artist_data.get('artistic_philosophy', {})

    def _extract_technical_info(self, entities):
        """Extract relevant technical information."""
        return self.artist_data.get('technical_expertise', {})

    def _extract_historical_info(self, entities):
        """Extract relevant historical information."""
        return self.artist_data.get('historical_context', {})

    def _extract_personal_info(self, entities):
        """Extract relevant personal information."""
        return self.artist_data.get('personal_life', {})

    def _generate_response(self, topic, content):
        """Generate a response based on the topic and content."""
        if not content:
            return "I'm not sure about that. Could you ask me something else?"
        
        # Add personality traits to the response
        traits = self.personality if self.personality else ["I"]
        emotional_context = random.choice(traits)
        
        if topic == 'identity':
            birth_info = f"I was born on {content['birth']}" if content['birth'] != 'unknown' else "I was born"
            return f"As {content['name']}, {emotional_context}. {birth_info}. {content.get('background', '')}"
        elif topic == 'artistic':
            education = content.get('education', {}).get('early_training', {})
            education_text = education.get('skills_acquired', ['I developed my artistic skills'])[0]
            return f"{emotional_context}. {education_text}"
        elif topic == 'philosophy':
            beliefs = content.get('core_beliefs', {}).get('nature_of_art', ['I have my own unique perspective on art'])[0]
            return f"{emotional_context}. {beliefs}"
        elif topic == 'technical':
            methods = content.get('methods', ['I use various techniques in my work'])[0]
            return f"{emotional_context}. {methods}"
        elif topic == 'historical':
            context = content.get('cultural_environment', 'My time was a fascinating period')
            return f"{emotional_context}. {context}"
        elif topic == 'personal':
            interests = content.get('interests', ['I have many interests'])[0]
            return f"{emotional_context}. {interests}"
        
        return "I'm not sure how to respond to that. Could you ask me something else?"

    def respond(self, query):
        """Generate a response to the user's query."""
        topic, entities = self._analyze_query(query)
        content = self._get_relevant_content(topic, entities)
        response = self._generate_response(topic, content)
        
        # Update conversation state
        self.conversation_focus = topic
        self.previous_topics.append(topic)
        
        return response

# Example usage
if __name__ == "__main__":
    # Initialize chatbot with path to local model
    model_path = "models/llama-2-7b-chat.gguf"  # Update this path to your local model
    chatbot = ArtistChatbot("data/kahlo.json")
    
    print("Chat initialized. Type 'exit' to end the conversation.")
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            break
        response = chatbot.respond(query)
        print(f"Artist: {response}") 