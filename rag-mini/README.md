# 📄 Mini RAG Q&A System

> A Retrieval-Augmented Generation (RAG) system that answers questions strictly from uploaded documents — zero hallucination by design.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![FAISS](https://img.shields.io/badge/FAISS-CPU-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.1-purple)

---

## Tech Stack

| Layer | Tool |
|-------|------|
| UI | Streamlit |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Vector DB | FAISS (in-memory, IndexFlatL2) |
| LLM | Groq API — LLaMA 3.1 8B Instant |
| Framework | LangChain |
| Language | Python 3.12 |

---

## How It Works

```
Document (TXT / PDF)
        │
        ▼
   Document Loader          ← PyPDFLoader / TextLoader
        │
        ▼
   Text Chunking            ← 400 words/chunk · 50-word overlap
        │
        ▼
   Embedding Generation     ← all-MiniLM-L6-v2 · batch_size=64 · normalized
        │
        ▼
   FAISS Vector Store       ← incremental batch indexing (100 chunks/batch)
        │
     [Query]
        │
        ▼
   Top-3 Retrieval          ← cosine similarity search
        │
        ▼
   LLM Answer Generation    ← LLaMA 3.1 8B via Groq · temperature=0
        │
        ▼
      Answer (grounded in document only)
```

---

## Setup

### 1. Clone and enter directory

```bash
git clone https://github.com/yourusername/rag-assessment.git
cd rag-assessment/rag-mini
```

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

```env
# .env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 5. Run

```bash
# Streamlit UI
streamlit run app.py

# CLI (alternative)
python cli.py
```

---

## Sample Test Queries

Using `sample_doc/company-policy.txt`:

| Question | Expected Answer |
|----------|----------------|
| How many paid leaves are allowed? | 20 days annually |
| What is the work from home policy? | Up to 2 days/week with manager approval |
| When are salaries credited? | Last working day of every month |
| What is the Friday dress code? | Casual |
| How long is maternity leave? | 26 weeks |
