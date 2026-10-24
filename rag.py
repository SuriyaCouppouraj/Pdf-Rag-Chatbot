"""
Beginner RAG: PDF Q&A with citations.

Usage:
    python rag.py ingest path/to/file.pdf
    python rag.py ask "What is the refund policy?"
"""
import os
import sys

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

CHUNK_SIZE = 800   # characters per chunk (try 400 / 800 / 1200 later)
OVERLAP = 150      # characters shared between neighbouring chunks
TOP_K = 4          # how many chunks to retrieve per question
MODEL = "claude-sonnet-5-5"

# 1) Vector database (saved to ./chroma_db so you only ingest once)
client = chromadb.PersistentClient(path="chroma_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # free, runs locally
)
collection = client.get_or_create_collection("docs", embedding_function=embed_fn)


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    step = size - overlap
    pieces = [text[i:i + size] for i in range(0, len(text), step)]
    return [p for p in pieces if p.strip()]


def ingest(pdf_path):
    name = os.path.basename(pdf_path)
    reader = PdfReader(pdf_path)
    ids, docs, metas = [], [], []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for j, piece in enumerate(chunk_text(text)):
            ids.append(f"{name}-p{page_num}-c{j}")
            docs.append(piece)
            metas.append({"source": name, "page": page_num})
    if not docs:
        print("No text found. Is this a scanned PDF? (needs OCR)")
        return
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"Ingested {len(docs)} chunks from {name}")


def ask(question):
    # 2) Retrieve the most relevant chunks
    results = collection.query(query_texts=[question], n_results=TOP_K)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        print("Nothing ingested yet. Run: python rag.py ingest file.pdf")
        return

    context = "\n\n".join(
        f"[{m['source']} p.{m['page']}]\n{d}" for d, m in zip(docs, metas)
    )

    # 3) Generate an answer using ONLY that context
    system = (
        "You answer questions using only the provided context. "
        "Cite sources like [file.pdf p.3]. "
        "If the answer is not in the context, reply exactly: "
        "\"I don't know based on the provided documents.\""
    )
    llm = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    response = llm.messages.create(
        model=MODEL,
        max_tokens=500,
        system=system,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    print(response.content[0].text)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("ingest", "ask"):
        print(__doc__)
        sys.exit(1)
    command, arg = sys.argv[1], " ".join(sys.argv[2:])
    ingest(arg) if command == "ingest" else ask(arg)