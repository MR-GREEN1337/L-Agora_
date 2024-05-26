import streamlit as st
from Graph import Graph
from langchain_core.messages import HumanMessage

# Define profile images for each persona
profile_images = {
    "aristotle": "https://www.wikiberal.org/images/4/49/Buste_d%27Aristote.jpg",
    "shopenhauer": "https://upload.wikimedia.org/wikipedia/commons/b/bc/Arthur_Schopenhauer_by_J_Sch%C3%A4fer%2C_1859b.jpg",
    "freud": "https://upload.wikimedia.org/wikipedia/commons/3/36/Sigmund_Freud%2C_by_Max_Halberstadt_%28cropped%29.jpg",
    "Hegel": "https://images.gr-assets.com/authors/1651531859p8/6188.jpg"
}

# Function to process the chat prompt and get the response
def get_chat_response(prompt):
    graph = Graph.kickoff()
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=prompt
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 30},
    )
    return events

# Streamlit app
st.title("L'Agora")

st.markdown("<p style='text-align: center; color: #808080; opacity: 0.7; margin: 0;'>``Thou Shall seek the truth``</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #808080; opacity: 0.7; margin: 0;'>``Thou Shall find``</p>", unsafe_allow_html=True)

# Center the content
st.markdown(
    """
    <style>
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .chat-container {
            width: 80%;
            max-width: 800px;
            margin-top: 2em;
        }
        .message {
            display: flex;
            align-items: flex-start;
            margin-bottom: 1em;
            padding: 1em;
            background: #f9f9f9;
            border-radius: 10px;
        }
        .message img {
            border-radius: 50%;
            margin-right: 1em;
        }
        .message strong {
            font-size: 1.1em;
            margin-bottom: 0.5em;
        }
        .message p {
            margin: 0;
        }
        .input-container {
            width: 80%;
            max-width: 800px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 1em;
        }
        .input-container input {
            flex: 1;
            padding: 0.5em;
            margin-right: 0.5em;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .input-container button {
            padding: 0.5em 1em;
            border-radius: 5px;
            border: none;
            background: #007bff;
            color: white;
            cursor: pointer;
        }
        .input-container button:hover {
            background: #0056b3;
        }
        .footer {
            text-align: center;
            font-size: 12px;
            color: #808080;
            margin-top: 1em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Container for the chat history
chat_history = st.container()

# Center the input box and button
with st.container():
    with st.form("prompt_form", clear_on_submit=True):
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        prompt = st.text_input("Initiate the debate:", "", key="prompt_input")
        submit_button = st.form_submit_button(label="Send")
        st.markdown('</div>', unsafe_allow_html=True)

# Define CSS styles for the chat history container
st.markdown(
    """
    <style>
        .chat-history {
            max-height: 300px; /* Adjust the max height as needed */
            overflow-y: auto; /* Enable vertical scrollbar */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if submit_button:
    if prompt:
        with st.spinner('Thinking...'):
            events = get_chat_response(prompt)
            for output in events:
                for key, value in output.items():
                    for message in value["messages"]:
                        sender = key
                        content = message.content
                        if sender != "call_tool" and len(content) != 0:
                            with chat_history:
                                profile_image = profile_images.get(sender, "")
                                st.markdown(f"""
                                <div class="chat-history">
                                    <div class="chat-container">
                                        <div class="message">
                                            <img src="{profile_image}" width="50">
                                            <div>
                                                <strong>{sender.title()}</strong>
                                                <p>{content}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
    else:
        st.error("Please enter a prompt.")


# Add the final image
st.image("https://c4.wallpaperflare.com/wallpaper/384/613/223/raphael-athens-philosophy-arch-wallpaper-preview.jpg", use_column_width=True)

# Footer
st.markdown('<div class="footer">© Mistral AI Hackathon 2024. All rights reserved.</div>', unsafe_allow_html=True)
