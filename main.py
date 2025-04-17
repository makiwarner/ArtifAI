import os
import json
from modules.retriever import load_retriever
from modules.fallback_rag import fetch_from_wikipedia
from modules.rewriter import rewrite_to_first_person
from modules.memory import ConversationMemory

retriever = load_retriever()
memory = ConversationMemory()

# Greeting and artist selection
def list_artists(artist_dir="artists"):
    return [f.replace(".json", "") for f in os.listdir(artist_dir) if f.endswith(".json")]

def choose_artist():
    print("Welcome to ArtifAI!\n")
    print("Available artists:")
    artists = list_artists()
    for i, name in enumerate(artists):
        print(f"[{i + 1}] {name}")
    while True:
        try:
            choice = int(input("\nChoose an artist by number: "))
            return artists[choice - 1]
        except (IndexError, ValueError):
            print("Invalid selection. Please try again.")

artist_name = choose_artist()
print(f"\nYou are now chatting with {artist_name}. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    query = f"{artist_name} {user_input}"
    docs = retriever.get_relevant_documents(query)

    if docs:
        response = docs[0].page_content
    else:
        fallback_docs = fetch_from_wikipedia(query)
        response = fallback_docs[0].page_content if fallback_docs else "I'm sorry, I don't have information on that."

    final_response = rewrite_to_first_person(response, artist_name)
    memory.add_turn(user_input, final_response)

    print(f"{artist_name}: {final_response}")