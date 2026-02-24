# AI Handbook Architect 🚀

A powerful AI-driven suite that generates professional, 20,000+ word handbooks and structured documentation from your PDF sources using GraphRAG and LongWriter techniques.

**🔗 [Live Demo](https://silver-ai-handbook.streamlit.app/)**

## ✨ Key Features

- **Knowledge Graph RAG**: Leverages LightRAG to build a deep understanding of your uploaded documents.
- **20k+ Word Generation**: Uses the LongWriter strategy to produce comprehensive, structured handbooks beyond standard LLM limits.
- **Interactive Drafting**: Chat with your documents to extract specific insights before or after generation.
- **Premium Interface**: A sleek, dark-mode Streamlit dashboard for a professional experience.

## 🛠️ Technology Stack

- **LLM**: Grok (Llama-3.3-70B) for high-reasoning output.
- **Indexing**: LightRAG for knowledge graph creation.
- **Vector DB**: Supabase (PostgreSQL) for persistent memory.
- **Frontend**: Streamlit.

## 🚀 How to Run the App

### 1. Cloud Deployment (Recommended)
The easiest way to view the app is via **Streamlit Community Cloud**:
- **Main File**: `app/main.py`
- **Python Version**: `3.11`
- **Secrets**: Ensure `GROK_API_KEY` and `POSTGRES_CONNECTION_STRING` are configured in the dashboard.

### 2. Local Setup
If you want to run the suite on your own machine:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tooba-Baqai/SilverAI-Assignment-AI-Engineering-main.git
   cd SilverAI-Assignment-AI-Engineering-main
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   Create a `.env` file in the `app/` directory with your API keys (see `.env.example`).
4. **Launch Application**:
   ```bash
   streamlit run app/main.py
   ```

## 📖 Usage Guide

1. **Upload**: Drag and drop your source PDF files into the sidebar.
2. **Index**: Click "Index Documents" to build the knowledge graph.
3. **Generate**: Ask the AI to "Generate a 20,000 word handbook based on these documents."
4. **Download**: Once finished, download your professional handbook in Markdown format.
