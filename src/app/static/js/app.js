document.addEventListener("DOMContentLoaded", function() {
    const chatForm = document.getElementById("chatForm");
    const messageInput = document.getElementById("messageInput");
    const chatContainer = document.getElementById("chat-container");

    chatForm.addEventListener("submit", function(e) {
        e.preventDefault();
        let message = messageInput.value.trim();
        if (message === "") return;
        addMessage("user", message);
        messageInput.value = "";

        // POST the user message to /chat
        fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // data.response = textual reply
            // data.avatar = e.g. "botticelli.jpg"
            addMessage("bot", data.response, data.avatar);
        })
        .catch(error => {
            console.error("Error:", error);
            addMessage("bot", "There was an error processing your request.");
        });
    });

    /**
     * Adds a new message to the chat container.
     * sender: "user" or "bot"
     * text: message text to display
     * avatar: optional image filename for the bot’s avatar
     */
    function addMessage(sender, text, avatar = null) {
        const messageElem = document.createElement("div");
        messageElem.classList.add("message", sender);

        // If it's a bot message and we have an avatar, display it
        if (sender === "bot" && avatar) {
            const imgElem = document.createElement("img");
            imgElem.src = `/static/images/${avatar}`;
            imgElem.classList.add("avatar");
            messageElem.appendChild(imgElem);
        }

        // Create a text span
        const textElem = document.createElement("span");
        textElem.textContent = text;
        messageElem.appendChild(textElem);

        chatContainer.appendChild(messageElem);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});

