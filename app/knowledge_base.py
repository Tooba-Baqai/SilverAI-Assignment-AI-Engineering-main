import os
import asyncio
import streamlit as st
import numpy as np
import nest_asyncio
from lightrag import LightRAG, QueryParam
from dotenv import load_dotenv

# Set up event loop for Streamlit
nest_asyncio.apply()
load_dotenv()

class FuncWrapper:
    """Wrapper to satisfy LightRAG's requirement for a .func attribute on callables."""
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class KnowledgeBase:
    def __init__(self, working_dir="./lightrag_storage"):
        self.working_dir = working_dir
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        
        # Database connection from Secrets
        pg_conn = st.secrets.get("POSTGRES_CONNECTION_STRING") or os.getenv("POSTGRES_CONNECTION_STRING")
        
        # Define the processing functions
        def llm_func(prompt, **kwargs):
            from handbook_generator import HandbookGenerator
            return HandbookGenerator()._get_completion(prompt)

        async def emb_func(texts):
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            if isinstance(texts, str):
                texts = [texts]
            return model.encode(texts)

        # Wrap functions to avoid AttributeError: .func
        wrapped_llm = FuncWrapper(llm_func)
        wrapped_emb = FuncWrapper(emb_func)

        try:
            if pg_conn:
                # Use Addon parameters for Postgres integration
                self.rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=wrapped_llm,
                    embedding_func=wrapped_emb,
                    kv_storage="PGKVStorage",
                    doc_status_storage="PGDocStatusStorage",
                    graph_storage="PGGraphStorage",
                    vector_storage="PGVectorStorage",
                    addon_params={"postgres_conn_config": pg_conn}
                )
            else:
                self.rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=wrapped_llm,
                    embedding_func=wrapped_emb
                )
        except Exception as e:
            # Fallback to local storage if anything fails
            st.warning(f"Storage initialization failed, falling back to local: {e}")
            self.rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=wrapped_llm,
                embedding_func=wrapped_emb
            )

    def insert_text(self, text):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.rag.ainsert(text))

    def query(self, query, mode="hybrid"):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.rag.aquery(query, QueryParam(mode=mode)))
