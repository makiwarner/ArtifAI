document.addEventListener("DOMContentLoaded", function() {
    const chatForm = document.getElementById("chatForm");
    const messageInput = document.getElementById("messageInput");
    const chatContainer = document.getElementById("chat-container");
    const currentArtistName = document.getElementById("current-artist-name");
    const currentArtistAvatar = document.getElementById("current-artist-avatar");

    // Constants for repeated strings
    const CHAT_ENDPOINT = "/chat";
    const CONTENT_TYPE = "Content-Type";

    function updateCurrentArtist(artistName, avatar) {
        if (artistName) {
            currentArtistName.textContent = artistName;
            if (avatar) {
                currentArtistAvatar.src = `/static/images/${avatar}`;
                currentArtistAvatar.style.display = "block";
            }
        } else {
            currentArtistName.textContent = "None";
            currentArtistAvatar.style.display = "none";
        }
    }

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
            addMessage("bot", data.response, data.avatar);
            
            // Update current artist if provided
            if (data.current_artist) {
                updateCurrentArtist(data.current_artist, data.avatar);
            }
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

