
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import numpy as np
import tempfile
import os

load_dotenv()

class LocalTfidfEmbeddings(Embeddings):

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(max_features=dim)
        self._is_fitted = False
        self._fit_texts = []

    def _fit_if_needed(self, texts: list[str]):
        all_texts = self._fit_texts + texts
        self.vectorizer = TfidfVectorizer(max_features=self.dim)
        self.vectorizer.fit(all_texts)
        self._fit_texts = all_texts
        self._is_fitted = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._fit_if_needed(texts)
        vectors = self.vectorizer.transform(texts).toarray()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        if not self._is_fitted:
            self._fit_if_needed([text])
        vector = self.vectorizer.transform([text]).toarray()[0]
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

st.set_page_config(
    page_title="Research Tool — PDF Q&A",
    page_icon="📄",
    layout="centered",
)

st.markdown("""
<style>
    .pdf-chip {
        display: inline-block;
        background: #1a1a2e;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 6px 14px;
        margin: 4px;
        font-size: 13px;
        color: #58a6ff;
    }
    .status-box {
        padding: 12px 18px;
        border-radius: 10px;
        font-size: 14px;
        margin-bottom: 12px;
    }
    .status-ready {
        background: rgba(63, 185, 80, 0.08);
        border: 1px solid rgba(63, 185, 80, 0.2);
        color: #3fb950;
    }
    .status-empty {
        background: rgba(210, 153, 34, 0.08);
        border: 1px solid rgba(210, 153, 34, 0.2);
        color: #d29922;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        streaming=True,
    )

model = get_llm()

def process_pdfs(uploaded_files):
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = uploaded_file.name

        all_docs.extend(pages)
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = LocalTfidfEmbeddings(dim=512)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(all_docs), len(chunks)


def get_relevant_context(vectorstore, query, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    context_parts = []
    sources = set()

    for doc in docs:
        context_parts.append(doc.page_content)
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        sources.add(f"{src} (p.{int(page) + 1})" if page != "?" else src)

    context = "\n\n---\n\n".join(context_parts)
    return context, sources

RAG_SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question based on the provided context from their PDF documents.

Rules:
- Base your answer ONLY on the provided context.
- If the context doesn't contain enough information, say so honestly.
- Cite which document and page the information came from when possible.
- Be concise but thorough.

Context from uploaded PDFs:
{context}"""

with st.sidebar:
    st.markdown("## 📄 Upload PDFs")
    st.caption("Upload one or more PDF files to query them with AI.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("🔄 Process PDFs", use_container_width=True, type="primary"):
            with st.spinner("Reading & embedding your PDFs..."):
                vectorstore, n_pages, n_chunks = process_pdfs(uploaded_files)
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_names = [f.name for f in uploaded_files]
                st.session_state.n_pages = n_pages
                st.session_state.n_chunks = n_chunks
                st.session_state.history = []
            st.success(f"✅ {n_pages} pages → {n_chunks} chunks indexed!")

    st.markdown("---")

    if "vectorstore" in st.session_state:
        st.markdown('<div class="status-box status-ready">✅ PDFs loaded & ready</div>', unsafe_allow_html=True)
        for name in st.session_state.pdf_names:
            st.markdown(f'<span class="pdf-chip">📄 {name}</span>', unsafe_allow_html=True)
        st.caption(f"{st.session_state.n_pages} pages · {st.session_state.n_chunks} chunks")
    else:
        st.markdown('<div class="status-box status-empty">⏳ No PDFs loaded yet</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.session_state.rag_mode = st.toggle(
        "📚 PDF-Aware Mode",
        value=True,
        help="ON = answers from your PDFs. OFF = normal chat."
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ── MAIN CHAT ─────────────────────────────────────────────────
st.header("📄 Research Tool")

if st.session_state.get("rag_mode") and "vectorstore" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to get started, or turn off **PDF-Aware Mode** to chat freely.")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📌 Sources"):
                for src in msg["sources"]:
                    st.caption(f"• {src}")

user_input = st.chat_input("Ask something about your PDFs...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    messages = []
    sources = set()

    use_rag = (
        st.session_state.get("rag_mode", True)
        and "vectorstore" in st.session_state
    )

    if use_rag:
        context, sources = get_relevant_context(
            st.session_state.vectorstore, user_input
        )
        system_msg = RAG_SYSTEM_PROMPT.format(context=context)
    else:
        system_msg = "You are a helpful research assistant. Answer the user's questions."

    messages.append(SystemMessage(content=system_msg))

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

        if use_rag and sources:
            with st.expander("📌 Sources"):
                for src in sources:
                    st.caption(f"• {src}")

    st.session_state.history.append({
        "role": "ai",
        "content": full_response,
        "sources": list(sources) if sources else [],
    })