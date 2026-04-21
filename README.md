# 🏥 MediAssist — RAG-Powered Healthcare Chatbot

> A fully offline, conversational AI healthcare assistant built with Retrieval-Augmented Generation (RAG), FAISS vector search, and lightweight local models. No API key, no internet required after setup.

---

## 🚀 What It Does

MediAssist answers healthcare questions accurately by combining:
1. **Semantic Search** — finds the most relevant medical knowledge for your question (FAISS)
2. **AI Answer Generation** — generates a clear, grounded response using a local LLM (Flan-T5)
3. **Multi-turn Memory** — remembers conversation context for follow-up questions
4. **REST API** — all features available as clean JSON endpoints

**Real-world use case:** Patient FAQ bot, hospital website assistant, pre-consultation tool, healthcare information kiosk.

---

## 🧠 RAG Architecture (How It Works)

```
User Question
     │
     ▼
[all-MiniLM-L6-v2]        ← Convert question to 384-dim vector (90MB)
     │
     ▼
[FAISS Index]              ← Find top-3 most similar medical documents
Top 3 Relevant Chunks
     │
     ▼
[Prompt Builder]           ← Combine: context + chat history + question
     │
     ▼
[google/flan-t5-base]      ← Generate grounded answer (250MB, CPU-fast)
     │
     ▼
JSON Response
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.9+, Flask |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (90MB) |
| Vector Store | FAISS CPU (in-memory, sub-10ms search) |
| LLM | `google/flan-t5-base` (250MB, 100% offline) |
| RAG Framework | Custom pipeline (LangChain-free, lightweight) |
| Frontend | HTML5, CSS3, Vanilla JS |

> **Total model size: ~340MB** — runs on any laptop with 4GB RAM, no GPU needed.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat` | Send message, get AI health answer |
| POST | `/api/v1/session/new` | Start new conversation session |
| GET | `/api/v1/session/<id>/history` | Get chat history |
| DELETE | `/api/v1/session/<id>/clear` | Clear session |
| POST | `/api/v1/search` | Raw semantic search (no LLM) |
| GET | `/api/v1/health` | Health check + model stats |

### Example

```bash
curl -X POST http://localhost:5002/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What causes high blood pressure?", "session_id": "test123"}'
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/mediassist.git
cd mediassist

# 2. Install
pip install -r requirements.txt

# 3. Run
python app.py

# 4. Open
# http://localhost:5002
```

> ⚠️ **First run:** Models download automatically (~340MB total). After that, everything runs offline instantly.  
> **No API key needed. No internet after setup.**

---

## 📁 Project Structure

```
mediassist/
├── app.py              # Flask app + full RAG pipeline
├── requirements.txt
└── templates/
    └── index.html      # Chat UI + API docs tab
```

---

## 🏥 Knowledge Base Coverage

25 medical topics including: Fever, Headache, Cough, Chest Pain, Diabetes, Hypertension, Asthma, Anemia, Thyroid, Paracetamol, Ibuprofen, Antibiotics, First Aid (cuts, burns), Stroke, Heart Attack, Depression, Anxiety, Sleep, Nutrition, Vitamin D, Iron, Dehydration, Allergies, Back Pain.

---

## ⚠️ Disclaimer

MediAssist is for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 👩‍💻 Author

**Deeksha Chowdhary** | B.Tech CSE (AI & ML), Malla Reddy University, Hyderabad  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/deekshachowdhary)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/YOUR_USERNAME)
