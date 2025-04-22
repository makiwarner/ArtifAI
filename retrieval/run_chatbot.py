from query_retriever import query_artists
from generate_answer import generate_response

def chat():
    print("\n🎨 Welcome to the Artist Discovery Chatbot! Type 'exit' to quit.")

    try:
        while True:
            user_input = input("\nAsk a question: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("👋 Exiting chatbot. Bye!")
                break

            results = query_artists(user_input, k=1)
            if not results:
                print("❌ No matching artist found.")
                continue

            top_artist, _ = results[0]
            response = generate_response(top_artist, user_input)
            print("\n🤖 Response:", response)

    except EOFError:
        print("\n🔚 Input ended.")
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user.")

if __name__ == "__main__":
    chat()
