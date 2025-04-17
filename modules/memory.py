class ConversationMemory:
    def __init__(self, max_length=5):
        self.history = []
        self.max_length = max_length

    def add_turn(self, user_input, bot_response):
        self.history.append({"user": user_input, "bot": bot_response})
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def get_context(self):
        return self.history