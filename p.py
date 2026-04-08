# ✅ Hide all warnings
import warnings
warnings.filterwarnings("ignore")

# ✅ Hide transformers logs
from transformers import logging
logging.set_verbosity_error()

# ✅ Hide HF hub logs
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader("sample_research_paper.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})
llm = ChatGroq(model="llama-3.1-8b-instant")
print("📄 Chatbot Ready!\n")

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        break

    results = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
You are an intelligent assistant.

Answer the question ONLY using the provided context.- Be clear and detailed.- If not found, say "Answer not found in document".
Context:
{context}
Question:
{query}
Answer:
"""
    response = llm.invoke(prompt)

    print("\n🤖", response.content)
  