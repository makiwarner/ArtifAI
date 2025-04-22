import os
import torch
import atexit
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

_tokenizer = None
_model = None

def cleanup_model():
    global _model, _tokenizer
    if _model is not None:
        _model.cpu()
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Register cleanup
atexit.register(cleanup_model)

def load_model():
    global _tokenizer, _model
    try:
        if _tokenizer is None or _model is None:
            if not os.path.exists(MODEL_DIR):
                raise FileNotFoundError(f"Model directory not found at {MODEL_DIR}")
                
            model_file = os.path.join(MODEL_DIR, "model.safetensors")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found at {model_file}")
                
            print("Loading tokenizer and model...")
            _tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
            _model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, local_files_only=True)
            _model.eval()
            print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    return _tokenizer, _model

def generate_response(artist_name, question):
    try:
        print(f"Generating response for artist: {artist_name}")
        print(f"Question: {question}")
        
        # Read artist data to get direct attributes
        artist_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed-bios.csv"))
        artist_info = artist_data[artist_data['name'] == artist_name].iloc[0]
        
        # Handle specific attribute questions directly
        question_lower = question.lower()
        
        # Handle general art movement questions
        if 'what is' in question_lower and any(term in question_lower for term in ['movement', 'style', 'impressionism', 'surrealism', 'cubism']):
            movements_mentioned = [mov for mov in ['impressionism', 'surrealism', 'cubism'] if mov in question_lower]
            if movements_mentioned:
                if movements_mentioned[0] in artist_info['genre'].lower():
                    return f"{movements_mentioned[0].title()} is an art movement that {artist_name} was associated with. {artist_info['bio']}"
                else:
                    return f"While {artist_name} was not associated with {movements_mentioned[0].title()}, they were known for {artist_info['genre']}."
        
        # Handle nationality questions
        if 'nationality' in question_lower or 'what country' in question_lower:
            return f"{artist_name} was {artist_info['nationality']}."
            
        # Handle movement/style questions
        if any(term in question_lower for term in ['movement', 'style', 'artistic movement', 'associated with']):
            movements = artist_info['genre'].split(',')
            if len(movements) == 1:
                return f"{artist_name} was associated with {movements[0].strip()}."
            elif len(movements) == 2:
                return f"{artist_name} was associated with {movements[0].strip()} and {movements[1].strip()}."
            else:
                movements = [m.strip() for m in movements]
                movements_str = ", ".join(movements[:-1]) + f", and {movements[-1]}"
                return f"{artist_name} was associated with {movements_str}."
                
        # Handle painting count questions
        if 'how many' in question_lower and ('paintings' in question_lower or 'works' in question_lower):
            count = artist_info['paintings']
            return f"{artist_name} has {count} known paintings/works attributed to them."
            
        # Handle birth/death date questions
        if any(term in question_lower for term in ['born', 'died', 'birth', 'death', 'when did', 'lifespan']):
            years = artist_info['years'].split(' - ')
            if len(years) == 2:
                birth, death = years
                return f"{artist_name} lived from {birth} to {death}."
            return f"{artist_name}'s lifespan was {artist_info['years']}."
            
        # Handle biographical questions
        if any(term in question_lower for term in ['tell me about', 'who is', 'who was', 'more about']):
            return artist_info['bio']
        
        # For other questions, use the model with improved context
        tokenizer, model = load_model()
        
        # Include relevant attributes in the prompt for context
        context = f"{artist_name} was a {artist_info['nationality']} artist ({artist_info['years']}) known for {artist_info['genre']}. {artist_info['bio']} "
        prompt = f"Based on this information about {artist_name}: {context}\nQuestion: {question}\nAnswer:"
        
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs=tokenizer.encode(prompt, return_tensors="pt"),
                max_length=200,
                do_sample=True,
                top_k=30,
                top_p=0.85,
                temperature=0.6,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("Answer:")[-1].strip()
        
        # Clean up the response
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        unique_sentences = []
        seen = set()
        for s in sentences:
            s_lower = s.lower()
            if s_lower not in seen:
                unique_sentences.append(s)
                seen.add(s_lower)
        
        response = '. '.join(unique_sentences)
        if not response.endswith('.'):
            response += '.'
            
        return response if response else artist_info['bio']
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error processing your request. Please try again."
