from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq model (make sure GROQ_API_KEY is set in your environment)
model = ChatGroq(model="llama-3.1-8b-instant")

# Initialize chat history
chat_history = [SystemMessage(content="You are a helpful assistant.")]

# First system and user messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain.")
]

# Add user messages to history
chat_history.extend(messages)

# Get AI response
result = model.invoke(chat_history)

# Append AI message to history
chat_history.append(AIMessage(content=result.content))

# Print entire chat history
for msg in chat_history:
    print(f"{msg.__class__.__name__}: {msg.content}")