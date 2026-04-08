import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import logging
logging.set_verbosity_error()

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot")

PDF_FOLDER = "PDF"

# ✅ Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔄 Load PDFs
if st.button("🔄 Load PDFs"):

    if not os.path.isdir(PDF_FOLDER):
        st.error(f"❌ Folder '{PDF_FOLDER}' not found")
        st.stop()

    pdf_files = [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        st.error("❌ No PDF files found!")
        st.stop()

    all_documents = []

    for file in pdf_files:
        loader = PyPDFLoader(file)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = os.path.basename(file)

        all_documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    docs = splitter.split_documents(all_documents)

    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    embeddings = load_embeddings()

    db = FAISS.from_documents(docs, embeddings)
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": 5})

    st.success(f"✅ Loaded {len(pdf_files)} PDFs successfully!")

# 💬 Input
with st.form("chat_form", clear_on_submit=True):
    query = st.text_input("💬 Ask a question:")
    submit = st.form_submit_button("Ask")

if submit and query:
    if "retriever" not in st.session_state:
        st.warning("⚠️ Please load PDFs first")
    else:
        results = st.session_state.retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in results])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in results]))

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        prompt = f"""Answer ONLY from the context.
If not found, say "Answer not found in document".

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        # ✅ Save chat
        st.session_state.chat_history.append({
            "question": query,
            "answer": response.content,
            "sources": sources
        })

# ✅ SHOW CHAT AS TOGGLE
for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
    with st.expander(f"💬 Question {i}: {chat['question']}", expanded=False):
        st.markdown("### 🤖 Answer")
        st.write(chat["answer"])

        st.markdown("### 📚 Sources")
        for src in chat["sources"]:
            st.write(f"- {src}")