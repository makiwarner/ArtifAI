from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.query_retriever import query_artists
from retrieval.generate_answer import generate_response

app = Flask(__name__, static_folder="static", template_folder="templates")

# Create a single instance of the chatbot
bot = None

def get_bot():
    global bot
    if bot is None:
        bot = WebArtistChatBot()
    return bot

class WebArtistChatBot:
    def __init__(self):
        self.end_chat = False
        self.current_artist = None
        self.conversation_history = []

    def respond(self, user_input):
        if user_input.lower() in ["bye", "quit", "exit"]:
            self.end_chat = True
            return "Goodbye!", None, None

        # If question is about previous artist (using pronouns or implicit references)
        question_lower = user_input.lower()
        has_pronouns = any(word in question_lower for word in ['he', 'she', 'they', 'his', 'her', 'their', 'them'])
        
        if self.current_artist and (has_pronouns or not any(name.lower() in question_lower for name in query_artists.artist_names)):
            # Use current artist context
            answer = generate_response(self.current_artist, user_input)
            avatar_filename = f"{self.current_artist.lower().replace(' ', '_')}.jpg"
            return answer, avatar_filename, self.current_artist

        # Otherwise try to find new artist context
        top_matches = query_artists(user_input, k=1)
        if not top_matches:
            # Keep current artist context even if no new artist is found
            return "Sorry, I couldn't find a relevant artist.", None, self.current_artist

        best_artist, _ = top_matches[0]
        self.current_artist = best_artist
        answer = generate_response(best_artist, user_input)
        avatar_filename = f"{best_artist.lower().replace(' ', '_')}.jpg"
        
        return answer, avatar_filename, best_artist


@app.route("/")
def index():
    # Initialize the bot when the page loads
    get_bot()
    greeting = "🎨 Welcome to ArtifAI! Ask a question about a famous artist."
    return render_template("index.html", greeting=greeting)


@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
            
        bot = get_bot()
        response, avatar, current_artist = bot.respond(user_input)
        
        if not response:
            return jsonify({"error": "Could not generate response"}), 500
            
        return jsonify({"response": response, "avatar": avatar, "current_artist": current_artist})
    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({"error": "There was an error processing your request."}), 500


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)


if __name__ == "__main__":
    app.run(debug=True)
