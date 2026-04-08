import streamlit as st, numpy as np, tempfile, os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, streaming=True)

class TfidfEmbeddings(Embeddings):
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=512)
        self.fitted = False

    def _norm(self, m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1
        return (m / n).tolist()

    def embed_documents(self, texts):
        self.vec.fit(texts)
        self.fitted = True
        return self._norm(self.vec.transform(texts).toarray())

    def embed_query(self, text):
        if not self.fitted:
            self.embed_documents([text])
        v = self.vec.transform([text]).toarray()[0]
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v).tolist()

def process_pdfs(files):
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        pages = PyPDFLoader(path).load()
        for p in pages:
            p.metadata["source"] = f.name
        docs.extend(pages)
        os.unlink(path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    return FAISS.from_documents(chunks, TfidfEmbeddings())

def get_context(vs, query):
    docs = vs.similarity_search(query, k=4)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = {
        f"{d.metadata.get('source', 'Unknown')} p.{d.metadata.get('page', '?')}"
        for d in docs
    }
    return context, sources

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF Q&A")

with st.sidebar:
    st.markdown("## Upload PDFs")
    files = st.file_uploader(
        "PDFs", type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if files and st.button("▶ Process PDFs", type="primary", use_container_width=True):
        with st.spinner("Indexing..."):
            st.session_state.vs = process_pdfs(files)
            st.session_state.history = []
        st.success(f"✅ {len(files)} file(s) ready!")
    if "vs" in st.session_state:
        st.caption("PDFs loaded and ready.")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

if "vs" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to get started.")

if "history" not in st.session_state:
    st.session_state.history = []

# Show chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📌 Sources"):
                for s in msg["sources"]:
                    st.caption(f"• {s}")

if q := st.chat_input("Ask something about your PDFs..."):
    with st.chat_message("user"):
        st.markdown(q)
    st.session_state.history.append({"role": "user", "content": q})

    sources = set()
    if "vs" in st.session_state:
        context, sources = get_context(st.session_state.vs, q)
        system_prompt = (
            f"Answer ONLY from context below. If not found, say so.\n\n{context}"
        )
    else:
        system_prompt = "You are a helpful assistant."

    msgs = [SystemMessage(content=system_prompt)] + [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in st.session_state.history
    ]

    with st.chat_message("assistant"):
        # FIX: guards against None chunks from Groq
        reply = st.write_stream(
            (chunk.content or "") for chunk in llm.stream(msgs)
        )
        if sources:
            with st.expander("📌 Sources"):
                for s in sources:
                    st.caption(f"• {s}")

    st.session_state.history.append({
        "role": "assistant",
        "content": reply,
        "sources": list(sources),
    })