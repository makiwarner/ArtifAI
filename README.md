# ArtifAI: An Interactive Art History Chatbot

ArtifAI is an intelligent chatbot that allows users to interact with and learn from artists throughout history. Using advanced natural language processing and machine learning techniques, it provides informative responses about artists, their works, and their historical context.

![ArtifAI Interface](https://github.com/user-attachments/assets/2fd73169-202f-49b5-a535-5ce410cadb52)

## Features

- 🎨 Interactive conversations about famous artists
- 🖼️ Context-aware responses maintaining conversation history
- 📚 Rich biographical information and artistic insights
- 🤖 Advanced semantic search using FAISS for accurate artist matching
- 🎯 Smart context switching between different artists
- 👤 Visual feedback with artist avatars

## How It Works

ArtifAI processes and responds to user queries through a sophisticated pipeline:

### 1. Data Collection Phase
- Scrapes artist biographical data from Wikipedia
- Collects artwork information from curated datasets
- Aggregates information about artistic movements and styles
- Stores raw data in structured CSV format

### 2. Data Preprocessing Pipeline
a. Text Cleaning
   - Removes HTML and special characters
   - Normalizes Unicode characters
   - Standardizes whitespace and formatting
   
b. Entity Extraction
   - Identifies key entities (people, places, dates)
   - Tags artistic movements and styles
   - Recognizes artwork references
   
c. Text Tokenization & Lemmatization
   - Breaks text into meaningful tokens
   - Reduces words to their base form
   - Removes stop words and noise

### 3. Feature Engineering
a. Text Vectorization
   - Generates TF-IDF matrices for text analysis
   - Creates semantic embeddings using Sentence Transformers
   - Produces dense vector representations of artist biographies

b. Embedding Generation
   - Uses state-of-the-art language models
   - Creates high-dimensional semantic representations
   - Optimizes for artist similarity matching

### 4. Model Training
a. Language Model Fine-tuning
   - Fine-tunes GPT-2 on art historical content
   - Trains on artist biographies and artwork descriptions
   - Optimizes for natural language generation

b. Response Generation Training
   - Develops context-aware response patterns
   - Trains on question-answer pairs
   - Incorporates artistic domain knowledge

### 5. Search and Retrieval
a. FAISS Implementation
   - Builds efficient similarity search index
   - Enables fast nearest-neighbor lookups
   - Optimizes for real-time response

b. Context Management
   - Maintains conversation history
   - Tracks current artist context
   - Enables natural follow-up questions

### 6. Web Interface
a. Real-time Processing
   - Handles user inputs dynamically
   - Processes queries in real-time
   - Maintains session context

b. Visual Feedback
   - Displays artist avatars
   - Shows current conversation context
   - Provides intuitive user interface

## Technical Architecture

The project is structured into several key components:

### Data Pipeline
- Data collection from various sources including Wikipedia
- Text preprocessing and cleaning
- Entity extraction and tokenization
- Feature generation using embeddings

### Machine Learning Components
- Sentence transformers for text embeddings
- FAISS index for efficient similarity search
- Fine-tuned GPT-2 model for response generation
- Custom-trained dialogue model for natural conversations

### Web Interface
- Flask-based web server
- Real-time chat interface
- Dynamic artist context display
- Responsive design with artist avatars

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ArtifAI.git
cd ArtifAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required model files:
- Place GloVe embeddings in `data/glove.6B.300d.txt`
- Download pre-trained models into the `output/` directory

## Usage

1. Start the Flask server:
```bash
python ui/app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Start chatting! You can ask questions about artists such as:
- "Tell me about Claude Monet"
- "What artistic movement was Salvador Dali associated with?"
- "How many paintings did Vincent van Gogh create?"

## Project Structure

```
ArtifAI/
├── data/                  # Data files and embeddings
├── pipeline/             # Data processing pipeline
│   ├── data_collection/  # Web scraping and data gathering
│   ├── data_preprocessing/ # Text cleaning and processing
│   ├── features/        # Feature extraction and embeddings
│   └── model/           # Model training and configuration
├── retrieval/           # Search and response generation
├── ui/                  # Web interface
│   ├── static/         # CSS, JS, and images
│   └── templates/      # HTML templates
└── evaluation/          # Model evaluation metrics
```

## Technologies Used

- Python 3.8+
- Flask for web server
- PyTorch and Transformers
- FAISS for similarity search
- Sentence Transformers
- Wikipedia API for data collection
- GPT-2 for response generation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the Hugging Face team for their transformers library
- Facebook Research for FAISS
- OpenAI for GPT-2
- All the amazing artists whose work and lives we learn from

