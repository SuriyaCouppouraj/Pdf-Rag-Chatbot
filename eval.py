"""
Evaluate the RAG pipeline on a fixed test set and compare chunk sizes.

    python eval.py Nimbus_Tech_Employee_Handbook.pdf          # retrieval only (fast, no API calls)
    python eval.py Nimbus_Tech_Employee_Handbook.pdf --full   # also test the LLM's answers (uses the API)
    python eval.py Nimbus_Tech_Employee_Handbook.pdf --k 1 --methods vector hybrid rerank   # compare retrieval methods

Metrics
  Retrieval hit : the chunk containing the correct fact was in the top-K results
  Answer acc    : the LLM's answer contains the correct value
  Refusal acc   : for questions NOT in the PDF, the bot says "I don't know"
"""
import argparse
import csv
import itertools
import os
import re
import time

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

MODEL = "gemini-3.8-flash"
CONFIGS = [(400, 80), (800, 150), (1200, 200)]  # (chunk_size, overlap)

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "Cite sources like [file.pdf p.3]. "
    "If the answer is not in the context, reply exactly: "
    "\"I don't know based on the provided documents.\""
)

# (question, regex for the fact in the source text, regex for a correct answer)
ANSWERABLE = [
    ("How many days of paid annual leave does a full-time employee get per year?", r"18 days of paid annual leave", r"\b18\b"),
    ("How many unused annual leave days can be carried forward?", r"maximum of 8 days", r"\b8\b"),
    ("How many days of paid sick leave do employees get per year?", r"10 days of paid sick leave", r"\b10\b"),
    ("After how many consecutive sick days is a medical certificate required?", r"exceeds 2 consecutive days", r"\b2\b|two"),
    ("How long is maternity leave?", r"26 weeks", r"\b26\b"),
    ("Within how many months of the birth must paternity leave be taken?", r"within 3 months", r"\b3\b|three"),
    ("How many working days in advance should planned leave longer than 3 days be requested?", r"at least 5 working days", r"\b5\b|five"),
    ("How long is the probation period?", r"probation period of 6 months", r"6 months|six months"),
    ("What is the notice period after confirmation?", r"After confirmation, the notice period is 60 days", r"\b60\b"),
    ("What is the notice period for Manager-level roles and above?", r"90 days", r"\b90\b"),
    ("How many days per week must employees work from the office?", r"3 days per week", r"\b3\b|three"),
    ("Which days are the required office days?", r"Tuesday, Wednesday, and Thursday", r"tuesday"),
    ("How many fully remote days can an employee request per year?", r"20 fully remote days", r"\b20\b"),
    ("How much is the one-time home-office allowance?", r"Rs\. 15,000", r"15,000|15000"),
    ("What is the monthly internet allowance for hybrid employees?", r"Rs\. 800", r"\b800\b"),
    ("What is the hotel limit per night in metro cities?", r"Rs\. 6,000 per night in metro", r"6,000|6000"),
    ("What is the daily food allowance during business travel?", r"Rs\. 1,200", r"1,200|1200"),
    ("Within how many days must expense claims be submitted?", r"within 30 days of the expense", r"\b30\b"),
    ("Within how many working days are approved claims paid?", r"within 10 working days", r"\b10\b"),
    ("What is the annual learning budget per employee?", r"Rs\. 25,000", r"25,000|25000"),
    ("How long must an employee stay after completing a company-sponsored certification?", r"at least 12 months", r"\b12\b"),
    ("In which months do performance reviews take place?", r"June and December", r"june"),
    ("How many points are on the performance rating scale?", r"5-point", r"\b5\b|five"),
    ("From which date do salary revisions take effect?", r"April 1", r"april"),
    ("What is the minimum password length?", r"at least 12 characters", r"\b12\b"),
    ("How often must passwords be changed?", r"every 90 days", r"\b90\b"),
    ("Within how many hours must a security incident be reported?", r"within 2 hours", r"\b2\b|two"),
    ("How many employees does Nimbus Tech have?", r"240 employees", r"\b240\b"),
    ("Who is the CEO of Nimbus Tech?", r"Anitha Raghavan", r"anitha"),
    ("How many public holidays does the company observe per year?", r"12 public holidays", r"\b12\b"),
]

# Questions whose answers are NOT in the PDF: the bot should refuse.
UNANSWERABLE = [
    "What is the company's stock price?",
    "Does Nimbus Tech offer free lunch in the cafeteria?",
    "Who is the CEO of Google?",
    "What is the dress code policy?",
    "What is the annual bonus percentage?",
]


def norm(text):
    return re.sub(r"\s+", " ", text)


def build_collection(client, ef, pdf_path, size, overlap):
    col = client.get_or_create_collection(f"cfg_{size}_{overlap}", embedding_function=ef)
    reader = PdfReader(pdf_path)
    name = os.path.basename(pdf_path)
    ids, docs, metas = [], [], []
    step = size - overlap
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for j, i in enumerate(range(0, len(text), step)):
            piece = text[i:i + size]
            if piece.strip():
                ids.append(f"{name}-p{page_num}-c{j}")
                docs.append(piece)
                metas.append({"source": name, "page": page_num})
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    return col


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class Retriever:
    """
    vector : plain embedding search (your current setup)
    hybrid : vector search + BM25 keyword search, merged with reciprocal rank fusion
    rerank : vector search for 10 candidates, then a cross-encoder re-scores them
    """
    POOL = 10  # candidates fetched before fusing / reranking

    def __init__(self, col, method, cross_encoder=None):
        self.col, self.method, self.ce = col, method, cross_encoder
        if method == "hybrid":
            from rank_bm25 import BM25Okapi
            data = col.get(include=["documents", "metadatas"])
            self.ids, self.docs, self.metas = data["ids"], data["documents"], data["metadatas"]
            self.bm25 = BM25Okapi([tokenize(d) for d in self.docs])

    def _vector(self, question, n):
        res = self.col.query(query_texts=[question], n_results=min(n, self.col.count()))
        return res["ids"][0], res["documents"][0], res["metadatas"][0]

    def search(self, question, k):
        if self.method == "vector":
            _, docs, metas = self._vector(question, k)
            return docs, metas
        if self.method == "rerank":
            _, docs, metas = self._vector(question, self.POOL)
            scores = self.ce.predict([(question, d) for d in docs])
            order = sorted(range(len(docs)), key=lambda i: -scores[i])[:k]
            return [docs[i] for i in order], [metas[i] for i in order]
        # hybrid: reciprocal rank fusion of the vector ranking and the BM25 ranking
        v_ids, _, _ = self._vector(question, self.POOL)
        bm_scores = self.bm25.get_scores(tokenize(question))
        bm_order = sorted(range(len(self.ids)), key=lambda i: -bm_scores[i])[: self.POOL]
        fused = {}
        for rank, id_ in enumerate(v_ids):
            fused[id_] = fused.get(id_, 0) + 1 / (60 + rank)
        for rank, i in enumerate(bm_order):
            fused[self.ids[i]] = fused.get(self.ids[i], 0) + 1 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        idx = {id_: i for i, id_ in enumerate(self.ids)}
        return [self.docs[idx[t]] for t in top], [self.metas[idx[t]] for t in top]


def generate(llm, types, question, docs, metas, retries=3):
    context = "\n\n".join(f"[{m['source']} p.{m['page']}]\n{d}" for d, m in zip(docs, metas))
    for attempt in range(retries):
        try:
            r = llm.models.generate_content(
                model=MODEL,
                contents=f"Context:\n{context}\n\nQuestion: {question}",
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            return r.text or ""
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"   API error ({str(e)[:90]}...) - retrying in {wait}s")
            time.sleep(wait)
    return ""


def pct(x, total):
    return f"{100 * x / total:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--full", action="store_true", help="also evaluate LLM answers (uses the API)")
    ap.add_argument("--k", type=int, default=4, help="chunks retrieved per question")
    ap.add_argument("--delay", type=float, default=5.0, help="seconds between API calls (free-tier limits)")
    ap.add_argument("--methods", nargs="+", default=["vector"], choices=["vector", "hybrid", "rerank"],
                    help="retrieval methods to compare")
    ap.add_argument("--chunk", type=int, default=None, help="only test this chunk size (e.g. 400)")
    args = ap.parse_args()

    llm, types = None, None
    if args.full:
        if "GEMINI_API_KEY" not in os.environ:
            raise SystemExit("GEMINI_API_KEY is not set. Set it first, or run without --full.")
        from google import genai
        from google.genai import types as gtypes
        llm, types = genai.Client(api_key=os.environ["GEMINI_API_KEY"]), gtypes

    client = chromadb.EphemeralClient()  # in-memory, does not touch your chroma_db folder
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    ce = None
    if "rerank" in args.methods:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    configs = [c for c in CONFIGS if args.chunk in (None, c[0])]
    n_ans, n_unans = len(ANSWERABLE), len(UNANSWERABLE)
    rows = []
    for (size, overlap), method in itertools.product(configs, args.methods):
        print(f"\n=== method={method}, chunk_size={size}, overlap={overlap} ===")
        col = build_collection(client, ef, args.pdf, size, overlap)
        retriever = Retriever(col, method, ce)
        hits = ans_ok = ref_ok = 0
        missed_retrieval, missed_answers = [], []

        for q, fact, ans in ANSWERABLE:
            docs, metas = retriever.search(q, args.k)
            hit = any(re.search(fact, norm(d), re.I) for d in docs)
            hits += hit
            if not hit:
                missed_retrieval.append(q)
            if args.full:
                reply = generate(llm, types, q, docs, metas)
                ok = bool(re.search(ans, norm(reply), re.I))
                ans_ok += ok
                if not ok:
                    missed_answers.append(q)
                time.sleep(args.delay)

        if args.full:
            for q in UNANSWERABLE:
                docs, metas = retriever.search(q, args.k)
                reply = generate(llm, types, q, docs, metas)
                ref_ok += bool(re.search(r"don.t know|do not know", reply, re.I))
                time.sleep(args.delay)

        print(f"chunks: {col.count()} | retrieval hit@{args.k}: {hits}/{n_ans}")
        for q in missed_retrieval:
            print(f"  retrieval missed: {q}")
        if args.full:
            print(f"answers correct: {ans_ok}/{n_ans} | refusals correct: {ref_ok}/{n_unans}")
            for q in missed_answers:
                print(f"  answer wrong: {q}")

        rows.append({
            "method": method, "chunk_size": size, "overlap": overlap, "chunks": col.count(),
            "retrieval_hit": pct(hits, n_ans),
            "answer_acc": pct(ans_ok, n_ans) if args.full else "-",
            "refusal_acc": pct(ref_ok, n_unans) if args.full else "-",
            "overall": pct(ans_ok + ref_ok, n_ans + n_unans) if args.full else "-",
        })

    print("\n================ SUMMARY ================")
    print(f"{'method':>7} {'chunk':>6} {'chunks':>7} {'retrieval':>10} {'answers':>8} {'refusals':>9} {'overall':>8}")
    for r in rows:
        print(f"{r['method']:>7} {r['chunk_size']:>6} {r['chunks']:>7} {r['retrieval_hit']:>10} "
              f"{r['answer_acc']:>8} {r['refusal_acc']:>9} {r['overall']:>8}")

    with open("eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved eval_results.csv")


if __name__ == "__main__":
    main()
