import streamlit as st
import requests
# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Chat",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Chat")

# -------------------------
# Session State
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Display Chat History
# -------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# User Input
# -------------------------
prompt = st.chat_input("Message")

if prompt:
    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    response = requests.post(
        "http://localhost:8000/chat",
        json={"prompt": prompt},
        stream=True,
    )

    def stream_response():
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")

    with st.chat_message("assistant"):
        assistant_response = st.write_stream(stream_response())

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
    )
