# 📚 IAS RAG Chatbot

Ask questions from your IAS/UPSC books using free AI models — optimised for low-spec laptops.

---

## ⚡ Quick Setup (5 minutes)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a FREE Groq API key
- Go to https://console.groq.com
- Sign up → Create API key (free, 14,400 requests/day)
- Copy the key (starts with `gsk_...`)

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Use the app
1. Upload your IAS PDFs (NCERT, Laxmikanth, etc.) in the sidebar
2. Click **Index uploaded books** (one-time per book)
3. Paste your Groq API key in the sidebar
4. Start asking questions!

---

## 💻 Low-spec laptop tips (LS5)

| Component | Recommended choice | Why |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | 90MB, CPU-only, fast |
| Vector DB | ChromaDB | No server needed |
| LLM | Groq free API | Cloud inference, no local GPU |
| Alt LLM | Ollama + Gemma 2B | Offline, needs ~4GB RAM |

---

## 🖥️ Using Ollama (offline, no internet)

```bash
# Install Ollama from https://ollama.com
ollama pull gemma:2b      # ~1.4GB download
ollama serve              # Keep running in background
```
Then select **Ollama (local)** in the sidebar.

---

## 📁 Project structure

```
ias_rag/
├── app.py            ← Streamlit UI
├── rag_engine.py     ← PDF loading, chunking, embeddings, retrieval
├── llm_connector.py  ← Groq + Ollama integrations
├── requirements.txt
├── chroma_db/        ← Auto-created: your vector database
└── README.md
```

---

## 📚 Recommended IAS books to add
- Indian Polity — M. Laxmikanth
- NCERT History (6–12)
- NCERT Geography (6–12)
- NCERT Economics (9–12)
- Indian Economy — Ramesh Singh
- Environment — Shankar IAS
