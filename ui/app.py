import os
import json
import re
import spacy
import nltk
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, jsonify, render_template
from modules.faq import get_best_faq_answer
from modules.sentence_matcher import get_best_sentence
from modules.flatten import flatten_content

nltk.download('stopwords')
nltk.download('punkt')

# Updated the ARTIST_FOLDER to use an absolute path for better compatibility
ARTIST_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'artists')
artist_files = [f for f in os.listdir(ARTIST_FOLDER) if f.endswith(".json")]
# Get sorted list of artist keys (filenames without .json)
artist_keys_sorted = sorted([os.path.splitext(f)[0] for f in artist_files])

# Global variable to hold the active chatbot instance (for a single session demo)
active_chatbot = None

app = Flask(__name__)

# ------------------------------------------------------------------------------
# NLP Helper class for the web-based artist chatbot using JSON/spaCy logic 
# ------------------------------------------------------------------------------
class WebArtistChatBot:
    """
    Loads an artist’s JSON from the artists folder and uses spaCy to find
    the best matching sentence or FAQ answer from the artist's data.
    """
    def __init__(self, json_filename, artist_directory="artists"):
        self.artist_directory = artist_directory
        self.artist_info = {}
        self.sentences = []
        self.faq_map = []  # List to store FAQ mappings: {"question": ..., "answer": ..., "notes": ...}
        self.nlp = spacy.load("en_core_web_md") 

        file_path = os.path.join(self.artist_directory, json_filename)
        with open(file_path, "r") as f:
            self.artist_info = json.load(f)

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
        return get_best_faq_answer(user_input, self.faq_map, self.nlp)

    def get_best_sentence(self, user_input):
        return get_best_sentence(user_input, self.sentences, self.nlp)

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

@app.route("/")
def index():
    """
    Render the main chat UI. The greeting message (welcome text with artists)
    is pre-computed and passed to the template.
    """
    greeting_msg = "Hello! Welcome to ArtifAI. Our supported artists are:<br>"
    artists = [
        "Hieronymus Bosch",
        "Sandro Botticelli",
        "Leonardo Da Vinci",
        "Frida Kahlo",
        "Gustav Klimt",
        "Claude Monet",
        "Pablo Picasso",
        "Diego Rivera",
        "Vincent Van Gogh",
        "Johannes Vermeer"
    ]
    for idx, name in enumerate(artists, start=1):
        greeting_msg += f"{idx}. {name}<br>"
    greeting_msg += "Who would you like to speak with today? (Type a number)"
    return render_template("index.html", greeting=greeting_msg)

@app.route("/chat", methods=["POST"])
def chat():
    global active_chatbot
    data = request.get_json() or {}
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please provide a message."})

    # If we haven't chosen an artist yet, the user must supply a valid integer:
    if not active_chatbot:
        try:
            index = int(user_input)
            if 1 <= index <= len(artist_keys_sorted):
                selected_key = artist_keys_sorted[index - 1]
                selected_file = f"{selected_key}.json"
                active_chatbot = WebArtistChatBot(selected_file, ARTIST_FOLDER)
                
                active_chatbot.avatar_filename = f"{selected_key}.jpg"
                artist_display_name = active_chatbot.artist_info.get("name", selected_key.title()).title()
                return jsonify({
                    "response": f"This is {artist_display_name}. What would you like to know?",
                    "avatar": active_chatbot.avatar_filename
                })
            else:
                return jsonify({"response": "Invalid number. Please try again."})
        except ValueError:
            # The user typed something that isn't an integer,
            # so re-prompt them to type an integer between 1 and N.
            return jsonify({"response": "Please type a valid integer corresponding to the artist you want to speak with."})
    else:
        # If the artist is already chosen, just chat freely with the bot.
        response_text = active_chatbot.respond(user_input)
        if active_chatbot.end_chat:
            active_chatbot = None
        return jsonify({
            "response": response_text,
            "avatar": getattr(active_chatbot, "avatar_filename", None)
        })

if __name__ == "__main__":
    app.run(debug=True)
