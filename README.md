# VerifAI

A Retrieval-Augmented Generation platform with built-in fact verification, confidence scoring, and self-calibration. Upload documents, ask questions with cited answers, verify claims, and compare pipeline modes side-by-side.

## Features

- **Q&A with Citations** — Ask questions against your knowledge base and get sourced answers with confidence scores
- **Fact Verifier** — Verify any claim against stored documents using entailment-based scoring
- **Pipeline X-Ray** — Side-by-side comparison of Full RAG, Basic RAG, and LLM-only responses
- **Knowledge Base Management** — Upload PDFs, seed from Wikipedia, search and manage documents
- **Dashboard** — Query history, confidence trends, and usage analytics
- **Self-Calibration** — Automatically adjusts retrieval and fetches new sources when confidence is low
- **Multi-Model Fallback** — Chains multiple LLM providers for reliability
- **Per-User Isolation** — Each user's uploads and history are isolated

## Architecture

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI |
| Frontend | HTML / CSS / JavaScript (SPA) |
| Embeddings | Sentence-BERT (local) |
| NLI Verification | DeBERTa (local) |
| Vector Store | ChromaDB (local) |
| LLM | Multi-provider fallback (Gemini, Groq) |
| Auth | MongoDB Atlas + JWT |

## Setup

### Prerequisites

- Python 3.10+
- MongoDB Atlas account (for authentication)
- At least one LLM API key (Gemini or Groq, both free tier)

### Installation

```bash
cd verifai
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your keys:

```
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
MONGODB_URI=your_mongodb_atlas_uri
JWT_SECRET=your_jwt_secret
```

At minimum, one LLM API key is required. `MONGODB_URI` enables user authentication; without it, the app runs in anonymous mode.

### Seed the Knowledge Base (Optional)

```bash
python -m scripts.seed_kb
```

You can also upload PDFs directly through the web interface.

### Run

```bash
python server.py
```

Open `http://localhost:5050` in your browser.
