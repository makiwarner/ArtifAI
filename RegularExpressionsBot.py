'''
This project builds an information retrieval (IR) chatbot 
that can get information from the "Best Artworks of All Time"
kaggle database, and scrape the Wikipedia page of a particular artist using 
BeautifulSoup in the topic of user's interest and collect information against 
user's queries following a heuristic backed by TF-IDF score and 
cosine-similarity score. This IR-ChatBot is user-friendly in permitting users 
to choose any artist to learn about, presenting either crisp and short response 
or detailed response. It leverages NLTK library to do text processing 
and scikit-learn library to do modeling. 
'''

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
import kagglehub #needed for dataset
import os #needed to make the dataset CSV
import re

#First, import the Kaggle Dataset
path = kagglehub.dataset_download("ikarus777/best-artworks-of-all-time")
csv_file = os.path.join(path, "artists.csv")
df = pd.read_csv(csv_file)
#print(df.head())

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
        "num_paintings": row['paintings']
    }

# ChatBot Class
class ArtistChatBot():
    
    def __init__(self):
        self.end_chat = False
        # self.got_topic = False
        # self.do_not_respond = True

        # Artist data storage
        self.artist_name = None
        self.artist_info = {}
        # self.wiki_text_data = []
        # self.sentences = []
        # self.para_indices = []
        # self.current_sent_idx = None
        
        # # NLP Tools
        # self.punctuation_dict = str.maketrans({p: None for p in punctuation})
        # self.lemmatizer = nltk.stem.WordNetLemmatizer()
        # self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
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
            print(f"ChatBot >> Sorry, {self.artist_info['name']} is not available to chat. Try another name.")

    #use regular expressions to answer basic info from the dataset.
    def respond(self, user_input):
        potential_inputs = {
            "birth|born|year": f"I was born in {self.artist_info['years'].split('-')[0]}.",
            "death|died|passed away": f"I passed away in {self.artist_info['years'].split('-')[-1]}.",
            "nationality|country": f"I am of {self.artist_info['nationality']} nationality.",
            "genre|style|movement|type|kind": f"I am most known for {self.artist_info['genre']} art.",
            "paintings|artworks|pieces": f"In total, I have created {self.artist_info['num_paintings']} documented paintings.",
            "bio|biography|about": f"Yes. Here is a short biography written about me: {self.artist_info['bio']}",
            "wiki|learn more|more information": f"You can read more about me here: {self.artist_info['wiki_link']}"
        }

        for potential_input, response in potential_inputs.items():
            if re.search(potential_input, user_input):
                print(f"{self.artist_info['name']} >> {response}")
                return
        
        print(f"{self.artist_info['name']} >> Sorry, I can only answer basic questions like birth year, nationality, and genre.")

if __name__ == '__main__':
    bot = ArtistChatBot()
    bot.chat()