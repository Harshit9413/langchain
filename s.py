from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import time

load_dotenv()

# ✅ Initialize model with streaming
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    streaming=True
)

# 🧠 Memory storage
chat_history = []

# 🎭 System personality
system_prompt = """
You are a smart, friendly AI assistant.

Rules:
- Give structured answers
- Use headings and bullet points
- Keep language simple
- Use emojis where useful
- Be clear and modern

Emoji Guide:
📌 Heading
👤 Person
🏢 Company
📅 Date
📍 Location
🚀 Achievements
⚠️ Important
✅ Conclusion
"""

def stream_output(text):
    """⚡ Typing effect like ChatGPT"""
    for char in text:
        print(char, end="", flush=True)
        time.sleep(0.01)   # speed control
    print()

def chat():
    print("🤖 Advanced AI Chatbot Started (type 'exit' to quit)\n")

    while True:
        user_input = input("👤 You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Goodbye!\n")
            break

        if not user_input:
            continue

        # 🧠 Add user message to memory
        chat_history.append(HumanMessage(content=user_input))

        # 📡 Send full conversation (memory + system)
        response = llm.invoke(
            [SystemMessage(content=system_prompt)] + chat_history
        )

        # 🧠 Store AI response
        chat_history.append(AIMessage(content=response.content))

        # 🎨 Output UI
        print("\n" + "═"*60)
        print("✨ AI RESPONSE:\n")

        # ⚡ Streaming effect
        stream_output(response.content)

        print("═"*60 + "\n")


if __name__ == "__main__":
    chat()