document.addEventListener("DOMContentLoaded", function() {
    const chatForm = document.getElementById("chatForm");
    const messageInput = document.getElementById("messageInput");
    const chatContainer = document.getElementById("chat-container");

    // Constants for repeated strings
    const CHAT_ENDPOINT = "/chat";
    const CONTENT_TYPE = "Content-Type";

    chatForm.addEventListener("submit", async function(e) {
        e.preventDefault();
        let message = messageInput.value.trim();
        if (message === "") return;
        addMessage("user", message);
        messageInput.value = "";

        try {
            // POST the user message to /chat
            const response = await fetch(CHAT_ENDPOINT, {
                method: "POST",
                headers: {
                    [CONTENT_TYPE]: "application/json"
                },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            // data.response = textual reply
            // data.avatar = e.g. "botticelli.jpg"
            addMessage("bot", data.response, data.avatar);
        } catch (error) {
            console.error("Error:", error);
            addMessage("bot", "There was an error processing your request.");
        }
    });

    /**
     * Adds a new message to the chat container.
     * @param {string} sender - The sender of the message (e.g., 'user' or 'bot').
     * @param {string} text - The message text.
     * @param {string} [avatar] - Optional avatar image for the bot.
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

