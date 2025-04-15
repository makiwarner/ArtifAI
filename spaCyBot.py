import json
import pandas as pd
from bs4 import BeautifulSoup
import requests
import nltk
from string import punctuation
import numpy as np
from time import sleep
import kagglehub  # Needed for dataset
import os  # Needed to make the dataset CSV
import re
import spacy  # Import spaCy

nltk.download('stopwords')
from nltk.corpus import stopwords

# First, import the Kaggle Dataset
path = kagglehub.dataset_download("ikarus777/best-artworks-of-all-time")
csv_file = os.path.join(path, "artists.csv")
df = pd.read_csv(csv_file)

# Parse the given info: load artist data from the dataset into the artist_data dictionary
artist_data = {}
for _, row in df.iterrows():
    artist_name = row['name'].strip().lower()
    artist_data[artist_name] = {
        "name": row['name'],
        "years": row['years'],
        "genre": row['genre'],
        "nationality": row['nationality'],
        "bio": row['bio'],
        "wiki_link": row['wikipedia'],
        "num_paintings": row['paintings'],
        "sentences": nltk.sent_tokenize(row['bio'])  # Tokenize bio into sentences
    }

# ChatBot Class using spaCy for semantic matching
class ArtistChatBot():
    
    def __init__(self):
        self.end_chat = False
        self.artist_name = None
        self.artist_info = {}
        # Load the spaCy model (ensure you have installed "en_core_web_md")
        self.nlp = spacy.load("en_core_web_md")
        
        # Greet User
        self.greeting()

    def greeting(self):
        print("Initializing ArtifAI ...")
        sleep(2)
        print('Type "bye", "quit", or "exit" to end chat')
        sleep(2)
        print('\nEnter an artist’s name to chat with them.')
        print('-' * 50)

    def chat(self):
        while not self.end_chat:
            user_input = input("User    >> ").strip().lower()
            if user_input in ["bye", "quit", "exit"]:
                print("ChatBot >> Goodbye! See you next time.")
                sleep(1)
                break
            elif not self.artist_name:
                self.process_artist(user_input)
            else:
                self.respond(user_input)

    def process_artist(self, artist_query):
        artist_query = artist_query.strip().lower()
        if artist_query in artist_data:
            self.artist_name = artist_query
            self.artist_info = artist_data[artist_query]
            print(f'{self.artist_info["name"]} >> Hello! I am {self.artist_info["name"]}. What would you like to know about me?')
        else:
            print("ChatBot >> Sorry, I don't know about that artist. Try another name.")

    def respond(self, user_input):
        potential_inputs = {
            "birth|born|year": f"I was born in {self.artist_info['years'].split('-')[0]}.",
            "death|died|passed away": f"I passed away in {self.artist_info['years'].split('-')[-1]}.",
            "nationality|country|from": f"I am of {self.artist_info['nationality']} nationality.",
            "genre|style|movement|type|kind": f"I am most known for {self.artist_info['genre']} art.",
            "paintings|artworks|pieces": f"In total, I have created {self.artist_info['num_paintings']} documented paintings.",
            "bio|biography|about": f"Yes. Here is a short biography written about me: {self.artist_info['bio']}",
            "wiki|learn more|more information": f"You can read more about me here: {self.artist_info['wiki_link']}"
        }

        # First, check direct keyword matches
        for potential_input, response in potential_inputs.items():
            if re.search(potential_input, user_input):
                print(f"{self.artist_info['name']} >> {response}")
                return
        
        # Use spaCy embeddings to match a relevant sentence in the bio
        best_sentence = self.get_best_sentence(user_input)
        if best_sentence:
            print(f"{self.artist_info['name']} >> {best_sentence}")
        else:
            print(f"{self.artist_info['name']} >> Sorry, I can only answer basic questions like birth year, nationality, and genre.")

    def get_best_sentence(self, user_input):
        """Find the most relevant sentence in the artist's biography using spaCy embeddings."""
        sentences = self.artist_info["sentences"]
        if not sentences:
            return None

        # Process the user input with spaCy
        user_doc = self.nlp(user_input)

        best_score = 0.0
        best_sentence = None

        # Evaluate each sentence in the biography
        for sentence in sentences:
            sentence_doc = self.nlp(sentence)
            score = user_doc.similarity(sentence_doc)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        # Return the sentence if the similarity is above a reasonable threshold (e.g., 0.4)
        if best_score > 0.4:
            return best_sentence

        return None

# Run the chatbot
if __name__ == '__main__':
    bot = ArtistChatBot()
    bot.chat()
