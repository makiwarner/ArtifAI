import os
import json
import re
import spacy
import nltk
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from .retriever import Retriever

nltk.download('stopwords')
nltk.download('punkt')

# Get the absolute path to the data directory
ARTIST_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

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

# Global variable to hold the active chatbot instance (for a single session demo)
active_chatbot = None

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.debug = True  # Enable debug mode

# Initialize spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Initialize retriever
retriever = Retriever()

# ------------------------------------------------------------------------------
# NLP Helper class for the web-based artist chatbot using JSON/spaCy logic 
# ------------------------------------------------------------------------------
class WebArtistChatBot:
    """
    Loads an artist's JSON from the artists folder and uses spaCy to find
    the best matching sentence or FAQ answer from the artist's data.
    """
    def __init__(self, json_filename, artist_directory="artists"):
        self.artist_directory = artist_directory
        self.artist_info = {}
        self.sentences = []
        self.faq_map = []  # List to store FAQ mappings: {"question": ..., "answer": ..., "notes": ...}
        
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download("en_core_web_md")
            self.nlp = spacy.load("en_core_web_md")

        file_path = os.path.join(self.artist_directory, json_filename)
        with open(file_path, "r") as f:
            self.artist_info = json.load(f)

        def flatten_content(item):
            """
            Recursively traverse the item and return a list of all string values found.
            """
            results = []
            if isinstance(item, str):
                results.append(item.strip())
            elif isinstance(item, list):
                for element in item:
                    results.extend(flatten_content(element))
            elif isinstance(item, dict):
                for key, value in item.items():
                    results.extend(flatten_content(value))
            return results

        # Process all info from the JSON.
        sections = [
            "bio", "lore", "knowledge", "influencesAndInfluenced", "techniqueAndMaterials",
            "interpretationsAndCriticism", "artworks/paintings", "FAQ", "postExamples", "topics",
            "style/kind", "adjectives"
        ]

        for section in sections:
            if section in self.artist_info:
                content = self.artist_info[section]
                # Special handling for FAQ to build our FAQ mapping.
                if section == "FAQ" and isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            question_text = item.get("question", "").strip()
                            answer_text = item.get("answer", "").strip()
                            notes_text = item.get("notes", "").strip() if item.get("notes") else ""
                            if question_text and answer_text:
                                self.faq_map.append({
                                    "question": question_text,
                                    "answer": answer_text,
                                    "notes": notes_text
                                })
                                # Optionally also add the answer to the general sentences.
                                self.sentences.append(answer_text)
                        elif isinstance(item, str):
                            self.sentences.append(item.strip())
                else:
                    # Use the recursive flatten_content to extract all strings.
                    flattened_sentences = flatten_content(content)
                    self.sentences.extend(flattened_sentences)

        # Remove duplicates if needed.
        self.sentences = list(set(self.sentences))
        self.end_chat = False


    def get_best_faq_answer(self, user_input):
        """
        Checks the FAQ mapping for the best matching question and returns its answer 
        if the similarity score is above the threshold.
        """
        if not self.faq_map:
            return None
        user_doc = self.nlp(user_input)
        best_score = 0.0
        best_answer = None
        for entry in self.faq_map:
            question_doc = self.nlp(entry["question"])
            score = user_doc.similarity(question_doc)
            if score > best_score:
                best_score = score
                best_answer = entry["answer"]
        if best_score > 0.6:
            return best_answer
        return None

    def get_best_sentence(self, user_input):
        """
        Fallback method for finding the best matching sentence from the general text.
        """
        if not self.sentences:
            return None
        user_doc = self.nlp(user_input)  #this function provides the NLP preprocessing and vectorization, using spaCy's "en_core_web_md" model
        best_score = 0.0
        best_sentence = None
        for sentence in self.sentences:
            sentence_doc = self.nlp(sentence)
            score = user_doc.similarity(sentence_doc)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        if best_score > 0.6:
            return best_sentence
        return None

    def respond(self, user_input):
        # Check for exit commands
        if user_input.lower() in ["bye", "quit", "exit"]:
            self.end_chat = True
            return "Goodbye! See you next time."

        # First, try matching against FAQ questions.
        faq_answer = self.get_best_faq_answer(user_input)
        if faq_answer:
            return faq_answer

        # Otherwise, use the general text matching.
        best_sentence = self.get_best_sentence(user_input)
        if best_sentence:
            return best_sentence
        else:
            return "I'm sorry, I don't have an answer for that."

# ------------------------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------------------------

@app.route('/')
def select_artist():
    """Display the artist selection screen."""
    print("Rendering artist selection page")
    artists = retriever.get_available_artists()
    return render_template('select.html', artists=artists)

@app.route('/handle_artist_selection', methods=['POST'])
def handle_artist_selection():
    """Handle artist selection and redirect to chat."""
    print("Handling artist selection")
    artist_id = request.form.get('artist_id')
    print(f"Selected artist ID: {artist_id}")
    
    if not artist_id:
        print("No artist ID provided")
        return jsonify({'error': 'No artist selected'}), 400
    
    session['artist_id'] = artist_id
    print(f"Stored artist_id in session: {session.get('artist_id')}")
    
    # Force session to be saved
    session.modified = True
    
    # Print all session data for debugging
    print(f"All session data: {dict(session)}")
    
    return redirect(url_for('chat'))

@app.route('/chat')
def chat():
    """Display the chat interface for the selected artist."""
    print("Rendering chat page")
    
    # Get artist_id from query parameters or session
    artist_id = request.args.get('artist_id') or session.get('artist_id')
    print(f"Artist ID from request/session: {artist_id}")
    
    if not artist_id:
        print("No artist_id found, redirecting to selection")
        return redirect(url_for('select_artist'))
    
    # Store in session for future requests
    session['artist_id'] = artist_id
    session.modified = True
    
    artist_name = retriever.get_artist_name(artist_id)
    print(f"Found artist name: {artist_name}")
    return render_template('chat.html', 
                         artist_id=artist_id,
                         artist_name=artist_name)

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handle chat messages and return responses."""
    artist_id = session.get('artist_id')
    if not artist_id:
        return jsonify({'error': 'No artist selected'}), 400
    
    message = request.json.get('message')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    response = retriever.get_response(artist_id, message)
    return jsonify({'success': True, 'response': response})

@app.route('/test_session')
def test_session():
    """Test if the session is working correctly."""
    print("Testing session")
    artist_id = session.get('artist_id')
    print(f"Current artist_id in session: {artist_id}")
    
    # Set a test value in the session
    session['test_value'] = 'test'
    session.modified = True
    
    # Print all session data for debugging
    print(f"All session data: {dict(session)}")
    
    return jsonify({
        'artist_id': artist_id,
        'test_value': session.get('test_value'),
        'all_session_data': dict(session)
    })

if __name__ == "__main__":
    app.run(debug=True)
