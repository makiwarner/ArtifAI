# ArtifAI - Chat with Famous Artists

ArtifAI is an interactive chatbot application that lets you have conversations with some of history's most renowned artists. Through advanced natural language processing and a rich knowledge base, each artist responds with their unique personality, sharing insights about their life, work, and artistic philosophy.

## Available Artists

- Sandro Botticelli
- Leonardo da Vinci
- Frida Kahlo
- Gustav Klimt
- Claude Monet
- Pablo Picasso
- Diego Rivera
- Vincent van Gogh
- Johannes Vermeer

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ArtifAI.git
cd ArtifAI
```

2. Create a virtual environment:
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

ArtifAI can be run in two modes: Command Line Interface (CLI) or Web Interface.

### CLI Mode

Run the application in CLI mode to chat with artists in your terminal:

```bash
python src/main.py --mode cli
```

In CLI mode:
1. You'll see a list of available artists
2. Enter the number corresponding to the artist you want to chat with
3. Start your conversation by typing your messages
4. Type 'exit', 'quit', or 'bye' to end the conversation

### Web Mode

Run the application in web mode to use the graphical interface in your browser:

```bash
python src/main.py --mode web
```

In web mode:
1. Open your browser and go to http://localhost:5000
2. Click on an artist's portrait to start a conversation
3. Use the chat interface to send messages and receive responses
4. Click the "End Chat" button or close the browser tab when you're done

## Features

- Natural language processing for understanding context and intent
- Personality-driven responses based on historical records
- Rich knowledge base covering each artist's:
  - Life and background
  - Artistic development
  - Technical expertise
  - Philosophical views
  - Historical context
  - Personal interests

## Requirements

- Python 3.8 or higher
- Flask for web interface
- spaCy for natural language processing
- NLTK for text processing
- Other dependencies listed in requirements.txt

## Project Structure

```
ArtifAI/
├── data/               # Artist JSON data files
├── src/               
│   ├── app/           # Web application files
│   ├── chatbot.py     # Core chatbot logic
│   └── main.py        # Application entry point
├── requirements.txt    # Python dependencies
└── README.md          # This file
```
