import os
import asyncio
import streamlit as st
import numpy as np
from lightrag import LightRAG, QueryParam
from dotenv import load_dotenv

load_dotenv()

class KnowledgeBase:
    def __init__(self, working_dir="./lightrag_storage"):
        self.working_dir = working_dir
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        
        # Pull connection string from Streamlit Secrets or Environment
        pg_conn = st.secrets.get("POSTGRES_CONNECTION_STRING") or os.getenv("POSTGRES_CONNECTION_STRING")
        
        # Force cache size to 0 for Supabase Pooler compatibility
        os.environ["POSTGRES_STATEMENT_CACHE_SIZE"] = "0"
        
        # Initialize LightRAG with the correct storage settings
        try:
            self.rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=self._llm_complete,
                embedding_func=self._embedding_func,
                kv_storage="PGKVStorage",
                doc_status_storage="PGDocStatusStorage",
                graph_storage="PGGraphStorage",
                vector_storage="PGVectorStorage",
                addon_conf={"postgres_conn_config": pg_conn}
            )
            print("DEBUG: Knowledge Base initialized successfully.")
        except Exception as e:
            print(f"ERROR: Initializing Knowledge Base: {e}")
            # Fallback to local storage if DB fails
            self.rag = LightRAG(working_dir=working_dir)

    def _llm_complete(self, prompt, **kwargs):
        # Local import to avoid circular dependencies
        from handbook_generator import HandbookGenerator
        gen = HandbookGenerator()
        return gen._get_completion(prompt)

    async def _embedding_func(self, texts):
        # Using a fast local embedding model to save API tokens
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        return embeddings

    def insert_text(self, text):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.rag.ainsert(text))

    def query(self, query, mode="hybrid"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.rag.aquery(query, QueryParam(mode=mode)))
