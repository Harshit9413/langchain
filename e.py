# chat_with_history.py
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# Store chat history
chat_history = []

def chat_with_history(user_input):
    """
    Takes user input, remembers previous messages, and returns LLM response.
    """
    # Add user message to history
    chat_history.append(HumanMessage(content=user_input))
    
    # Call LLM with all previous messages
    response = llm.invoke(chat_history)
    
    # Add AI response to history
    chat_history.append(AIMessage(content=response.content))
    
    return response.content

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    response = chat_with_history(user_input)
    print("AI:", response)