from functools import partialmethod

from tqdm import tqdm
from transformers import logging as tf_logging

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
tf_logging.set_verbosity_error()
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=".env")
loader = PyPDFLoader("sample_research_paper.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)
print("📄 Chatbot Ready!\n")
while True:
    query = input("Ask: ")
    if query.lower() == "exit":
        break

    results = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"""You are an intelligent assistant.
Answer the question ONLY using the provided context.- Be clear and detailed. If not found, say "Answer not found in document".
Context:
{context}
Question:
{query}
Answer:"""
    response = llm.invoke(prompt)
    print("\n🤖", response.content)
    print("\n" + "─" * 60 + "\n")