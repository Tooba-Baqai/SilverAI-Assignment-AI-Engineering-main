# 📚 LunarTech AI: Handbook Generator & Knowledge Graph RAG

This project is a high-performance AI Engineering application built for the **LunarTech AI Engineering Apprenticeship Assignment**. It features a Graph-based RAG system for document chat and an advanced **AgentWrite (LongWriter)** engine capable of generating comprehensive 20,000-word handbooks from PDF context.

## 🚀 Key Features
- **PDF Knowledge Extraction**: Uses `pdfplumber` and `LightRAG` to transform static PDFs into a dynamic Knowledge Graph.
- **GraphRAG Chat**: Context-aware chat interface that retrieves information from the Graph database to provide high-fidelity answers.
- **20k-Word Handbook Generator**: Implements the **AgentWrite (LongWriter)** technique (Plan -> Sequential Write) to bypass standard LLM output limits and generate massive, structured documents.
- **Asynchronous Engine**: A dedicated background threading model ensures the UI remains responsive even during heavy AI processing.
- **Resilient Generation**: Automatic rate-limit handling and model fallback (70B to 8B) for uninterrupted production-grade performance.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **AI Engine**: LightRAG (Graph-based Retrieval Augmented Generation)
- **Database**: Supabase (PostgreSQL + pgvector)
- **LLM Provider**: Groq (Llama 3.3 70B & Llama 3.1 8B)
- **Embeddings**: Sentence-Transformers (Local fallback) / OpenAI

## 📦 Setup Instructions

### 1. Prerequisites
- Python 3.10+
- A Supabase project with a PostgreSQL connection string.
- A Groq API Key.

### 2. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
pip install sentence-transformers nest_asyncio lightrag-hku
```

### 3. Environment Variables
Create a `.env` file in the `app/` directory:
```env
GROK_API_KEY=your_groq_api_key
POSTGRES_CONNECTION_STRING=your_supabase_connection_string
MODEL_PROVIDER=grok
MODEL_NAME=llama-3.3-70b-versatile
# Optional: OPENAI_API_KEY=your_key (If omitted, local embeddings will be used)
```

### 4. Running the Application
```bash
cd app
python -m streamlit run main.py
```

## 📖 Usage Guide
1. **Upload**: Drag and drop 1 or more PDFs into the sidebar.
2. **Process**: Click "Process Documents" to build the Knowledge Graph.
3. **Chat**: Ask questions about your documents in the chat window.
4. **Generate Handbook**: Type "Generate a 20,000-word handbook about [topic]" to start the LongWriter engine.

## 🧠 Methodology: AgentWrite (LongWriter)
This application implements the **LongWriter (AgentWrite)** methodology. Instead of requesting a long document in a single prompt (which leads to "lazy" outputs), the system:
1. **Retrieves Context**: Pulls relevant entity relationships from the Knowledge Graph.
2. **Plans**: Uses the LLM to architect a 20-section detailed outline.
3. **Writes**: Executes 20 sequential recursive prompts, maintaining state and context across sections to produce a cohesive 15,000 - 20,000 word document.

---
**Confidential — LunarTech AI Engineering Assignment**
*Developed by Tooba Baqai*
