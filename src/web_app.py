import streamlit as st
import requests

BACKEND_URL = "http://chatbot-container:5000/answer_message"

st.title("Pirate Weather Forecast (Cesena)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Put form first
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Insert your message here")
    submitted = st.form_submit_button("Send")

# Handle the new message before showing all messages
if submitted and user_input.strip():
    st.session_state.chat_history.append(user_input)
    response = requests.post(
        BACKEND_URL,
        json={"history": st.session_state.chat_history, "last_message": user_input}
    )
    st.session_state.chat_history.append(response.json().get("reply", ""))
    
# Now display the updated conversation
for idx, message in enumerate(st.session_state.chat_history):
    speaker = "You" if idx % 2 == 0 else "Bot"
    st.write(f"**{speaker}:** {message}")