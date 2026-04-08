import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    streaming=True  
)

st.header(" Research Tool")

if "history" not in st.session_state:
    st.session_state.history = []


for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

user_input = st.chat_input("Enter your prompt...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    messages = []
    for msg in st.session_state.history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        full_response = ""
        placeholder = st.empty()
        for chunk in model.stream(messages):
            full_response += chunk.content
            placeholder.markdown(full_response + "▌")  
        placeholder.markdown(full_response)  

    st.session_state.history.append({"role": "ai", "content": full_response})