# LunarTech Handbook Generator 🚀

This application generates comprehensive, 20,000-word handbooks from uploaded PDF documents using Advanced RAG and the LongWriter technique.

## Features
- **PDF Upload**: Parse and index multiple research papers/documentation.
- **Knowledge Graph RAG**: Uses LightRAG (Entity-Relationship Graph) for deep context retrieval.
- **Supabase Integration**: Persistent vector storage using Supabase pgvector.
- **Ultra-Long Generation**: Implements the **AgentWrite** technique to generate structured content exceeding typical LLM output limits.
- **Chat Interface**: Interactive Q&A and handbook request system.

## Setup Guide

### 1. Prerequisites
- Python 3.10+
- A Supabase project with `pgvector` enabled.
- API keys for Grok (X.AI), Gemini, or OpenAI.

### 2. Installation
```bash
cd app
pip install -r requirements.txt
```

### 3. Configuration
Rename `.env.example` to `.env` and fill in your credentials:
```env
# Example .env
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
POSTGRES_CONNECTION_STRING=postgresql://postgres:[password]@[host]:5432/postgres
```

### 4. Running the Application
```bash
streamlit run main.py
```

## How it Works
1. **Ingestion**: PDFs are parsed using `pdfplumber`. Text is chunked and entities are extracted to build a Knowledge Graph in LightRAG.
2. **Contextual Chat**: Use the chat interface to ask questions. LightRAG performs hybrid search (Vector + Graph) to provide accurate answers.
3. **Handbook Generation**:
   - **Phase 1 (Plan)**: The AI creates a detailed 50+ section outline based on the corpus.
   - **Phase 2 (Write)**: The AI writes each section incrementally, pulling specific context for each part and maintaining a "memory" of previous sections to ensure flow and avoid redundancy.

## Submission Details
- **Developer**: Antigravity (AI Engineering Apprentice Candidate)
- **Tech Stack**: Streamlit, LightRAG, Supabase, Grok/Gemini/GPT-4o.
