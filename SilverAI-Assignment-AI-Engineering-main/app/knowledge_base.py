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
        
        # Define the core functions to reuse
        llm_func = self._llm_complete
        emb_func = self._embedding_func

        # Initialize LightRAG
        try:
            if pg_conn:
                self.rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=llm_func,
                    embedding_func=emb_func,
                    kv_storage="PGKVStorage",
                    doc_status_storage="PGDocStatusStorage",
                    graph_storage="PGGraphStorage",
                    vector_storage="PGVectorStorage",
                    addon_conf={"postgres_conn_config": pg_conn}
                )
                print("DEBUG: Knowledge Base initialized with PostgreSQL.")
            else:
                raise ValueError("No database connection string found.")
        except Exception as e:
            print(f"WARNING: Database connection failed, falling back to local: {e}")
            # IMPORTANT: Re-pass the functions to the local fallback!
            self.rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=llm_func,
                embedding_func=emb_func
            )

    def _llm_complete(self, prompt, **kwargs):
        from handbook_generator import HandbookGenerator
        # Note: We don't pass arguments here to let it use st.secrets inside __init__
        gen = HandbookGenerator()
        return gen._get_completion(prompt)

    async def _embedding_func(self, texts):
        from sentence_transformers import SentenceTransformer
        # Download usually takes ~30s on first run in cloud
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
