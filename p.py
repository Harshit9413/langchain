import os
os.environ["TQDM_DISABLE"] = "1"
from tqdm import tqdm
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=".env")
pdf_files = ["machine_learning_notes.pdf", "sample_research_paper.pdf"]
all_documents = []
for file in pdf_files:
    loader = PyPDFLoader(file)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file
    all_documents.extend(docs)
documents = all_documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})2
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
print("📄 Chatbot Ready! (Multi-PDF Mode)\n")
while True:
    query = input("Ask: ")
    if query.lower() == "exit":
        break
    results = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in results])
    sources = set([doc.metadata.get("source", "Unknown") for doc in results])
    prompt = f"""You are an intelligent assistant.Answer the question ONLY using the provided context.- Be clear and detailed.- If not found, say "Answer not found in document".
Context:
{context}
Question:
{query}
Answer:"""
    response = llm.invoke(prompt)
    print("\n🤖", response.content)
    print("\n📚 Sources:", ", ".join(sources))
    print("\n" + "─" * 60 + "\n")