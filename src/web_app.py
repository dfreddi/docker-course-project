import streamlit as st
import requests

BACKEND_URL = "http://chatbot_container:8001/answer_message"

st.title("Pirate Weather Forecast (Cesena)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Insert your message here")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    response = requests.post(
        BACKEND_URL,
        json={"history": st.session_state.chat_history, "last_message": user_input}
    )
    st.session_state.chat_history.append(user_input)
    st.session_state.chat_history.append(response.json().get("answer", ""))
    
for idx, message in enumerate(st.session_state.chat_history):
    speaker = "You" if idx % 2 == 0 else "Bot"
    st.write(f"**{speaker}:** {message}")