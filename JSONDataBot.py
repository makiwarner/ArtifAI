import json
import os
import spacy
import nltk
import re
from time import sleep

# Download NLTK stopwords if needed (even if not used directly, this ensures proper NLTK setup)
nltk.download('stopwords')

class ArtistChatBot:
    def __init__(self, artist_directory="artists"):
        """
        Initializes the chatbot by setting the directory containing artist JSON files,
        loading the spaCy model, and prompting the user for an artist.
        """
        self.artist_directory = artist_directory
        self.artist_info = None
        self.sentences = []  # Will hold preprocessed sentences from the artist data
        self.nlp = spacy.load("en_core_web_md")  # Ensure you have this model installed
        self.end_chat = False
        
        self.load_artist()
        self.greeting()

    def load_artist(self):
        """
        Prompts the user to input an artist's name and loads the corresponding JSON file.
        It extracts the last word of the input as the key for the JSON file.
        It then preprocesses text data from the 'bio', 'lore', and 'knowledge' sections.
        """
        artist_input = input("Enter artist's name: ").strip().lower()
        # Extract the last word as the key (e.g., "Diego Rivera" -> "rivera")
        words = re.findall(r'\w+', artist_input)
        artist_key = words[-1] if words else artist_input
        file_name = f"{artist_key}.json"
        file_path = os.path.join(self.artist_directory, file_name)

        while not os.path.exists(file_path):
            print(f"Artist file '{file_name}' not found in '{self.artist_directory}'. Please try again.")
            artist_input = input("Enter artist's name: ").strip().lower()
            words = re.findall(r'\w+', artist_input)
            artist_key = words[-1] if words else artist_input
            file_name = f"{artist_key}.json"
            file_path = os.path.join(self.artist_directory, file_name)
        
        with open(file_path, "r") as f:
            self.artist_info = json.load(f)
        
        # Preprocess text: compile sentences from available sections and clean them
        sections = ["bio", "lore", "knowledge"]
        for section in sections:
            if section in self.artist_info and isinstance(self.artist_info[section], list):
                # Remove extra whitespace and ignore non-string entries
                self.sentences.extend([s.strip() for s in self.artist_info[section] if isinstance(s, str)])
        
        # Remove duplicates, if any
        self.sentences = list(set(self.sentences))

    def greeting(self):
        """
        Greets the user and provides basic instructions.
        """
        print('Type "bye", "quit", or "exit" to end chat.')
        sleep(2)
        artist_display_name = self.artist_info.get("name", "Unknown Artist").title()
        print(f"\nChatting with {artist_display_name}")
        print('-' * 50)
        print(f"{artist_display_name} >> Hello! What would you like to know about me?")


    def chat(self):
        """
        Main chat loop.
        """
        while not self.end_chat:
            user_input = input("User >> ").strip()
            if user_input.lower() in ["bye", "quit", "exit"]:
                print("ChatBot >> Goodbye! See you next time.")
                sleep(1)
                break
            else:
                self.respond(user_input)

    def respond(self, user_input):
        """
        Uses spaCy to find the best matching sentence from the preprocessed data.
        """
        best_sentence = self.get_best_sentence(user_input)
        artist_display_name = self.artist_info.get("name", "Artist").title()
        if best_sentence:
            print(f"{artist_display_name} >> {best_sentence}")
        else:
            print(f"{artist_display_name} >> I'm sorry, I don't have an answer for that.")

    def get_best_sentence(self, user_input):
        """
        Computes the similarity between the user query and each sentence from the artist data.
        Returns the sentence with the highest similarity score if it passes the threshold.
        """
        if not self.sentences:
            return None
        
        user_doc = self.nlp(user_input)
        best_score = 0.0
        best_sentence = None
        
        for sentence in self.sentences:
            sentence_doc = self.nlp(sentence)
            score = user_doc.similarity(sentence_doc)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        # Adjust the threshold based on your testing; 0.4 is a starting point.
        if best_score > 0.6:
            return best_sentence
        return None

if __name__ == "__main__":
    bot = ArtistChatBot()
    bot.chat()
