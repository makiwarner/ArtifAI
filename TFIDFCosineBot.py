import json 
import pandas as pd 
from bs4 import BeautifulSoup
import requests
import nltk
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from time import sleep
import kagglehub  # Needed for dataset
import os  # Needed to make the dataset CSV
import re

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

# ChatBot Class
class ArtistChatBot():
    
    def __init__(self):
        self.end_chat = False
        self.artist_name = None
        self.artist_info = {}

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
        
        # Use TF-IDF and Cosine Similarity to match a relevant sentence in the bio
        best_sentence = self.get_best_sentence(user_input)
        if best_sentence:
            print(f"{self.artist_info['name']} >> {best_sentence}")
        else:
            print(f"{self.artist_info['name']} >> Sorry, I can only answer basic questions like birth year, nationality, and genre.")

    def get_best_sentence(self, user_input):
        """Find the most relevant sentence in the artist's biography using TF-IDF and Cosine Similarity."""
        sentences = self.artist_info["sentences"]
        if not sentences:
            return None

        # ✅ **Step 1: Expand Keywords**
        synonyms = {
            "wife": ["married", "spouse", "partner", "husband"],
            "marriage": ["relationship", "wedding", "spouse", "husband", "wife"],
            "death": ["died", "passed away", "deceased"],
            "birth": ["born", "birthday", "birthdate"],
            "paintings": ["artworks", "pieces", "masterpieces"],
            "style": ["genre", "movement", "artistic style"]
        }

        # Expand user input if it contains key words
        for key, words in synonyms.items():
            if any(word in user_input.lower() for word in words):
                user_input += " " + key  # Append main keyword

        # ✅ **Step 2: Preprocess Input (Remove Stopwords & Punctuation)**
        stop_words = set(stopwords.words('english'))
        user_input_cleaned = ' '.join(
            [word for word in re.sub(r'[^\w\s]', '', user_input).lower().split() if word not in stop_words]
        )

        # ✅ **Step 3: Vectorize Sentences & Compute Similarity**
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences + [user_input_cleaned])
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

        # ✅ **Step 4: Find the Best Sentence**
        best_match_idx = cosine_similarities.argsort()[0][-1]  # Get the highest-ranked sentence
        best_score = cosine_similarities[0, best_match_idx]

        # ✅ **Step 5: Adjust Threshold**
        if best_score > 0.05:  # Reduce threshold to allow more responses
            return sentences[best_match_idx]

        return None

# Run the chatbot
if __name__ == '__main__':
    bot = ArtistChatBot()
    bot.chat()
