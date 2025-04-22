import os
import json
import signal
import sys
import subprocess
import argparse
import nltk
import spacy
from app.retriever import Retriever
from app.fallback_rag import fetch_from_wikipedia
from app.rewriter import rewrite_to_first_person
from app.memory import ConversationMemory
from app.app import app
from chatbot import ArtistChatbot

retriever = Retriever()
memory = ConversationMemory()

# Get the absolute path to the data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def check_spacy_model():
    """Check if the required spaCy model is installed."""
    try:
        spacy.load("en_core_web_sm")
        return True
    except OSError:
        return False

def install_spacy_model():
    """Install the required spaCy model."""
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def setup_nltk():
    """Download required NLTK data."""
    required_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            nltk.download(package)

def signal_handler(sig, frame):
    print("\nGoodbye! Thanks for chatting!")
    sys.exit(0)

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def get_available_artists():
    """Get list of available artist JSON files."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    artists = [f.replace('.json', '') for f in os.listdir(data_dir) if f.endswith('.json')]
    return artists

def run_cli_mode():
    """Run the chatbot in CLI mode."""
    # Check and install required NLP components
    if not check_spacy_model():
        print("Installing required spaCy model...")
        install_spacy_model()
    
    print("Checking NLTK data...")
    setup_nltk()
    
    # Get available artists
    artists = get_available_artists()
    
    if not artists:
        print("Error: No artist data files found in the data directory.")
        sys.exit(1)
    
    # Let user select an artist
    print("\nAvailable artists:")
    for i, artist in enumerate(artists, 1):
        print(f"{i}. {artist.replace('_', ' ').title()}")
    
    while True:
        try:
            choice = int(input("\nSelect an artist (enter number): "))
            if 1 <= choice <= len(artists):
                selected_artist = artists[choice - 1]
                break
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Please enter a valid number.")

    # Initialize chatbot with selected artist
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'{selected_artist}.json')
    chatbot = ArtistChatbot(data_path)
    
    # Welcome message
    artist_name = selected_artist.replace('_', ' ').title()
    print(f"\nWelcome! You are now chatting with {artist_name}.")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.\n")
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"\n{artist_name}: Farewell! Thank you for our conversation.")
                break
            
            # Get chatbot response
            if user_input:
                response = chatbot.respond(user_input)
                print(f"\n{artist_name}: {response}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{artist_name}: Goodbye! It was a pleasure speaking with you.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")

def run_web_mode():
    """Run the application in web mode."""
    print("Starting ArtifAI web server...")
    print("Open your browser to http://localhost:5000")
    app.run(debug=True)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Artist Chatbot')
    parser.add_argument('--mode', choices=['cli', 'web'], default='cli',
                      help='Run mode: cli for command line interface, web for web interface')
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        run_cli_mode()
    else:
        run_web_mode()

if __name__ == "__main__":
    main()