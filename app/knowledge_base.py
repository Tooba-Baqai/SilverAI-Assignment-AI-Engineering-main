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
        os.environ["POSTGRES_STATEMENT_CACHE_SIZE"] = "0"
        
        # Define functions
        def llm_func(prompt, **kwargs):
            from handbook_generator import HandbookGenerator
            return HandbookGenerator()._get_completion(prompt)

        async def emb_func(texts):
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(texts)

        try:
            if pg_conn:
                self.rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=llm_func,
                    embedding_func=emb_func, # Pass the function directly as most versions expect
                    kv_storage="PGKVStorage",
                    doc_status_storage="PGDocStatusStorage",
                    graph_storage="PGGraphStorage",
                    vector_storage="PGVectorStorage",
                    addon_conf={"postgres_conn_config": pg_conn}
                )
            else:
                self.rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=llm_func,
                    embedding_func=emb_func
                )
        except TypeError:
            # If positional arguments fail, use explicit keyword arguments
            # Some versions of LightRAG changed the signature
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
            else:
                self.rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=llm_func,
                    embedding_func=emb_func
                )
        except Exception as e:
            st.error(f"Initialization Error: {e}")
            # Final fallback
            self.rag = LightRAG(working_dir=working_dir, llm_model_func=llm_func, embedding_func=emb_func)

    def insert_text(self, text):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.rag.ainsert(text))

    def query(self, query, mode="hybrid"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.rag.aquery(query, QueryParam(mode=mode)))
