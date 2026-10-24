"""
Streamlit UI for the PDF Q&A chatbot (dark theme).
Run with:  streamlit run app.py
"""
import html
import time

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
import llm_client
from pypdf import PdfReader

CHUNK_SIZE = 800
OVERLAP = 150
TOP_K = 4
POOL = 10  # candidates fetched before reranking

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "Cite sources like [file.pdf p.3]. "
    "If the answer is not in the context, reply exactly: "
    "\"I don't know based on the provided documents.\""
)

SUGGESTIONS = [
    "Summarize the main points of this document",
    "Which dates or deadlines are mentioned?",
    "What numbers or amounts are mentioned?",
]

st.set_page_config(page_title="Docs Q&A", page_icon="📄", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Sora:wght@600;700&display=swap');
:root { --bg:#0F1522; --surface:#161E2E; --raised:#1D2740; --line:#263147;
        --text:#E6EBF5; --muted:#8D9AB5; --accent:#8B7CF6; --page:#F2C46D; }
html, body, .stApp, button, textarea, input { font-family:'Manrope', system-ui, sans-serif; }
.stApp { background: radial-gradient(1100px 560px at 75% -8%, #1b2547 0%, var(--bg) 58%); }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { display:none !important; }
.block-container { padding-top:2.4rem; max-width:820px; }
section[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--line); }

.hero h1 { font-family:'Sora', sans-serif; font-weight:700; font-size:2.4rem;
           letter-spacing:-0.02em; margin:0 0 .4rem; color:var(--text); }
.hero p { color:var(--muted); margin:0 0 1.6rem; font-size:1.02rem; max-width:34rem; }

[data-testid="stChatMessage"] { background:var(--surface); border:1px solid var(--line);
    border-radius:18px; padding:1rem 1.15rem; margin-bottom:.9rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background:var(--raised); border-color:#34416b; }
[data-testid="stChatInput"] { border-radius:16px; border:1px solid var(--line); background:var(--surface); }
[data-testid="stChatInput"]:focus-within { border-color:var(--accent); }

.stButton > button { background:var(--raised); border:1px solid var(--line); color:var(--text);
    border-radius:12px; text-align:left; transition:border-color .15s; }
.stButton > button:hover { border-color:var(--accent); color:#fff; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:transparent; }

.pill { display:inline-block; padding:.15rem .65rem; border:1px solid var(--line);
    border-radius:999px; color:var(--muted); font-size:.78rem; margin:.5rem .4rem .2rem 0; }
.src { border:1px solid var(--line); border-left:3px solid var(--page); border-radius:10px;
    padding:.7rem .9rem; margin:.5rem 0; background:#121A2B; }
.badge { background:var(--page); color:#1b1503; border-radius:6px; padding:.05rem .5rem;
    font-weight:700; font-size:.78rem; margin-right:.55rem; }
.src .file { color:var(--muted); font-size:.82rem; }
.src p { margin:.45rem 0 0; color:#C5CEE2; font-size:.88rem; line-height:1.55; }
.doc { padding:.55rem .75rem; border:1px solid var(--line); border-radius:10px;
    background:var(--bg); margin:.35rem 0; font-size:.86rem; word-break:break-all; }
.stat { color:var(--muted); font-size:.85rem; margin:.3rem 0 .9rem; }

/* Force readable colors even if the theme file is not loaded */
.stApp { color: var(--text); }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label, [data-testid="stExpander"] summary * { color: var(--text) !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
.doc { color: var(--text); }
[data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--raised) !important; border: 1px dashed var(--line) !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--surface) !important; border: 1px solid var(--line) !important; }
[data-testid="stAlert"] * { color: var(--text) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# Cached so the models load only once (not on every message)
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection("docs", embedding_function=embed_fn)


@st.cache_resource
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


collection = get_collection()


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    step = size - overlap
    pieces = [text[i:i + size] for i in range(0, len(text), step)]
    return [p for p in pieces if p.strip()]


def ingest_pdf(uploaded_file):
    name = uploaded_file.name
    reader = PdfReader(uploaded_file)
    ids, docs, metas = [], [], []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for j, piece in enumerate(chunk_text(text)):
            ids.append(f"{name}-p{page_num}-c{j}")
            docs.append(piece)
            metas.append({"source": name, "page": page_num})
    if docs:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def list_sources():
    metas = collection.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metas})


def answer_question(question, use_rerank=True, on_stage=None):
    t0 = time.time()
    n = POOL if use_rerank else TOP_K
    results = collection.query(query_texts=[question], n_results=min(n, collection.count()))
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if use_rerank:
        scores = get_reranker().predict([(question, d) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: -scores[i])[:TOP_K]
        docs = [docs[i] for i in order]
        metas = [metas[i] for i in order]
    context = "\n\n".join(
        f"[{m['source']} p.{m['page']}]\n{d}" for d, m in zip(docs, metas)
    )
    t1 = time.time()
    if on_stage:
        on_stage("Asking the model...")
    reply = llm_client.generate(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}")
    t2 = time.time()
    return reply, list(zip(docs, metas)), (t1 - t0, t2 - t1)


def friendly_error(e):
    msg = str(e)
    if "503" in msg or "UNAVAILABLE" in msg:
        return "The model is busy right now. Wait a minute, then ask again."
    if "504" in msg or "DEADLINE_EXCEEDED" in msg or "imed out" in msg or "imeout" in msg:
        return "The model took too long to answer. Try again in a moment."
    if "429" in msg:
        return "The free-tier limit was reached. Wait a minute, then ask again."
    if "404" in msg:
        return "This model name is not available. Copy a current model name from Google AI Studio (or console.groq.com) and set LLM_MODEL."
    return f"Something went wrong: {msg[:200]}"


def render_extras(srcs, timing, reranked):
    if timing:
        pills = (f'<span class="pill">Search {timing[0]:.1f}s</span>'
                 f'<span class="pill">{(llm_client.provider() or "Model").title()} {timing[1]:.1f}s</span>')
        if reranked:
            pills += '<span class="pill">Reranked</span>'
        st.markdown(pills, unsafe_allow_html=True)
    if srcs:
        with st.expander(f"Sources ({len(srcs)})"):
            cards = "".join(
                f'<div class="src"><span class="badge">p.{m["page"]}</span>'
                f'<span class="file">{html.escape(m["source"])}</span>'
                f'<p>{html.escape(t[:420])}{"…" if len(t) > 420 else ""}</p></div>'
                for t, m in srcs
            )
            st.markdown(cards, unsafe_allow_html=True)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### Documents")
    use_rerank = st.toggle("Better answers (reranking)", value=True)
    st.caption(f"Model: {llm_client.model_name() or 'not set'}")
    files = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process PDFs", use_container_width=True) and files:
        for f in files:
            with st.spinner(f"Reading {f.name}..."):
                n = ingest_pdf(f)
            if n:
                st.success(f"{f.name}: {n} passages added")
            else:
                st.warning(f"{f.name}: no text found. Scanned PDFs need OCR first.")

    sources = list_sources()
    if sources:
        st.markdown(f'<div class="stat">{len(sources)} document(s), {collection.count()} passages</div>',
                    unsafe_allow_html=True)
        st.markdown("".join(f'<div class="doc">{html.escape(s)}</div>' for s in sources),
                    unsafe_allow_html=True)
        if st.button("Remove all documents", use_container_width=True):
            ids = collection.get()["ids"]
            if ids:
                collection.delete(ids=ids)
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("No documents yet. Add a PDF to begin.")

# ---------------- Main ----------------
st.markdown(
    '<div class="hero"><h1>Ask your documents</h1>'
    '<p>Upload PDFs, then ask questions in plain language. Every answer shows the page it came from.</p></div>',
    unsafe_allow_html=True,
)

if llm_client.provider() is None:
    st.error(llm_client.key_help())
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    if sources:
        st.caption("Try asking")
        for s in SUGGESTIONS:
            if st.button(s, key=f"sug_{s}", use_container_width=True):
                st.session_state.pending = s
                st.rerun()
    else:
        st.info("Add a PDF in the sidebar, then click Process PDFs.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_extras(msg.get("sources"), msg.get("timing"), msg.get("reranked"))

question = st.chat_input("Ask a question about your documents") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        srcs, timing = [], None
        if not sources:
            reply = "Add and process a PDF first, then ask again."
            st.markdown(reply)
        else:
            try:
                with st.status("Searching your documents...") as status:
                    reply, srcs, timing = answer_question(
                        question, use_rerank, lambda s: status.update(label=s))
                    status.update(label="Answer ready", state="complete")
                st.markdown(reply)
                render_extras(srcs, timing, use_rerank)
            except Exception as e:
                reply = friendly_error(e)
                st.error(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": srcs,
         "timing": timing, "reranked": use_rerank}
    )