# 📚 LunarTech AI: Handbook Generator (AI Engineering Assignment)

A production-ready AI application that transforms PDF documents into a structured **Knowledge Graph (LightRAG)** and generates **20,000-word handbooks** using the **AgentWrite (LongWriter)** technique.

---

## 🚀 Executive Summary
This application solves the challenge of generating ultra-long contextually aware documents. By combining **LightRAG** for deep knowledge retrieval and a multi-stage **AgentWrite** execution pipeline, the system can produce cohesive handbooks exceeding 15,000+ words from user-uploaded PDFs.

## 🛠️ Technology Stack
- **Frontend**: Streamlit (Sleek, responsive chat interface)
- **AI Engine**: [LightRAG-hku](https://github.com/HKU-ADS/LightRAG) (Advanced GraphRAG)
- **LLM**: Groq (Llama-3.3-70B-Versatile)
- **Database**: Supabase (PostgreSQL with `pgvector` for vector storage)
- **PDF Core**: `pdfplumber` & `pypdf`
- **Embedding**: `sentence-transformers` (Local fallback for free/offline usage)

## 📖 Features & Functionality
- **Dynamic PDF Ingestion**: Extract and index complex PDF structures into a Knowledge Graph.
- **Graph-Based Retrieval**: Contextual chat that understands entity relationships across multiple documents.
- **20k Handbook Engine**: Implements the **LongWriter** methodology:
    1. **Planning Phase**: Architecting a 20-30 section detailed outline.
    2. **Writing Phase (Recursive)**: Sequential generation of 800-1000 word chapters, maintaining state through parent context.
- **Resilient AI Loop**: Built-in logic to handle API Rate Limits (429) by automatically sleeping and resuming.

## 💻 Setup & Installation

### 1. Prerequisites
- Python 3.10+
- [Supabase](https://supabase.com/) project with a DB connection string.
- [Groq](https://console.groq.com/) API Key.

### 2. Install Dependencies
```bash
pip install -r app/requirements.txt
```

### 3. Environment Configuration
Create an `.env` file in the `app/` directory:
```env
GROK_API_KEY=your_groq_key
POSTGRES_CONNECTION_STRING=your_supabase_pooler_url
MODEL_PROVIDER=grok
MODEL_NAME=llama-3.3-70b-versatile
```

### 4. Run the App
```bash
cd app
python -m streamlit run main.py
```

## 🏗️ Technical Approach & Challenges
### The "Different Event Loop" Challenge
Streaming heavy AI operations within Streamlit often results in `RuntimeError: bound to a different event loop`. I solved this by implementing a **Singleton Background Thread** architecture. This ensures the AI Engine lives in its own persistent loop, isolated from Streamlit's frequent UI reruns.

### Hitting the 20k Word Target
Standard LLMs suffer from "output starvation" after ~1000 words. Following the **AgentWrite** research, I implemented a plan-then-write sequence. The system first creates a comprehensive map, then writes each section while feeding the previous section's summary back into the prompt to ensure transition consistency.

---
**Developed for the LunarTech AI Engineering Apprenticeship.**
*GitHub Access granted to: vahekaren@gmail.com*
