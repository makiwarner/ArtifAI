class ConversationMemory:
    def __init__(self, max_length=5):
        self.history = []
        self.max_length = max_length

    def add_turn(self, user_input, bot_response, confidence=None, intent=None, emotional_context=None):
        """Add a conversation turn to the memory.
        
        Args:
            user_input: The user's input message
            bot_response: The bot's response
            confidence: Optional confidence score for the response
            intent: Optional detected intent of the user query
            emotional_context: Optional emotional context of the conversation
        """
        turn = {
            "user": user_input,
            "bot": bot_response
        }
        
        # Add optional parameters if provided
        if confidence is not None:
            turn["confidence"] = confidence
        if intent is not None:
            turn["intent"] = intent
        if emotional_context is not None:
            turn["emotional_context"] = emotional_context
            
        self.history.append(turn)
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def get_context(self):
        return self.history