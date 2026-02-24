import os
import asyncio
import streamlit as st
import numpy as np
import nest_asyncio
from lightrag import LightRAG, QueryParam
from dotenv import load_dotenv

# Apply nest_asyncio to handle Streamlit's event loop
nest_asyncio.apply()
load_dotenv()

class KnowledgeBase:
    def __init__(self, working_dir="./lightrag_storage"):
        self.working_dir = working_dir
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        
        # Pull connection string from Streamlit Secrets or Environment
        pg_conn = st.secrets.get("POSTGRES_CONNECTION_STRING") or os.getenv("POSTGRES_CONNECTION_STRING")
        os.environ["POSTGRES_STATEMENT_CACHE_SIZE"] = "0"
        
        # Define LLM function
        def llm_func(prompt, **kwargs):
            from handbook_generator import HandbookGenerator
            return HandbookGenerator()._get_completion(prompt)

        # Define Embedding function
        async def emb_func(texts):
            from sentence_transformers import SentenceTransformer
            # Use small, fast model for embedding
            model = SentenceTransformer('all-MiniLM-L6-v2')
            if isinstance(texts, str):
                texts = [texts]
            return model.encode(texts)

        # LIGHTRAG INITIALIZATION
        # Using a very defensive approach to handle constructor signature changes
        init_kwargs = {
            "working_dir": working_dir,
            "llm_model_func": llm_func,
            "embedding_func": emb_func
        }

        # If Supabase is connected, add storage params
        if pg_conn:
            init_kwargs.update({
                "kv_storage": "PGKVStorage",
                "doc_status_storage": "PGDocStatusStorage",
                "graph_storage": "PGGraphStorage",
                "vector_storage": "PGVectorStorage",
                "addon_conf": {"postgres_conn_config": pg_conn}
            })

        try:
            # Try keyword arguments (Standard)
            self.rag = LightRAG(**init_kwargs)
        except TypeError as e:
            # If standard fails, try falling back to basic setup
            print(f"DEBUG: LightRAG TypeError falling back: {e}")
            try:
                # Basic setup without storage override
                self.rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=llm_func,
                    embedding_func=emb_func
                )
            except Exception as final_e:
                st.error(f"Critical LightRAG Error: {final_e}")
                raise final_e

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
