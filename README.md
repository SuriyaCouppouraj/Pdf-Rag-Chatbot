# PDF Q&A Chatbot (RAG)

Ask questions about your PDFs in plain language. Every answer cites the page it came from, and the bot says "I don't know" when the document doesn't contain the answer.

## Demo

(Add your demo video here.)

## Features

- Upload several PDFs and chat with them in a dark-themed Streamlit app
- Page-level citations and a Sources panel that shows the exact passages used
- Refuses to answer when the answer is not in the documents
- Optional cross-encoder reranking for more accurate retrieval
- Works with Gemini or Groq (choose one by setting an API key)
- Evaluation script that measures retrieval accuracy on a 30-question test set

## How it works

1. Each PDF is split page by page into overlapping 800-character passages.
2. Passages are embedded with `all-MiniLM-L6-v2` and stored in Chroma.
3. A question retrieves the 10 closest passages.
4. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks them and keeps the best 4.
5. The LLM answers using only those passages and cites the pages.

## Evaluation

Tested on a 4-page sample handbook (`Nimbus_Tech_Employee_Handbook.pdf`) with 30 questions. A question counts as a hit if the passage containing the answer is the top-1 retrieved passage.

| Method | 400-char chunks | 800-char chunks | 1200-char chunks |
|---|---|---|---|
| Vector search | 93% | 87% | 87% |
| Hybrid (BM25 + vector) | 90% | 87% | 90% |
| Vector + reranking | 97% | 97% | 97% |

Reranking raised top-1 retrieval accuracy from 87% to 97%. Hybrid search did not help on this small document.

Limitations: this is one small document and the questions were written by me, so treat the numbers as indicative, not definitive.

Reproduce:

```bash
python eval.py Nimbus_Tech_Employee_Handbook.pdf --k 1 --methods vector hybrid rerank
```

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/pdf-rag-chatbot.git
cd pdf-rag-chatbot
pip install -r requirements.txt
```

Create a file named `.env` in the project folder:

```
GEMINI_API_KEY=your-key
LLM_MODEL=exact-model-name
```

To use Groq instead, set `GROQ_API_KEY` in `.env`. Run `python list_models.py` to see which Gemini models your key can use.

Start the app:

```bash
streamlit run app.py
```

## Tech stack

Python, Streamlit, ChromaDB, sentence-transformers, pypdf, Gemini or Groq.