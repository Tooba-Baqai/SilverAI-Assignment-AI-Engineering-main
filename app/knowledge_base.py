import os
import asyncio
import streamlit as st
import numpy as np
import nest_asyncio
import re
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv

# Set up event loop for Streamlit
nest_asyncio.apply()
load_dotenv()

class KnowledgeBase:
    def __init__(self, working_dir="./lightrag_storage"):
        self.working_dir = working_dir
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        
        # Database connection from Secrets
        pg_conn = st.secrets.get("POSTGRES_CONNECTION_STRING") or os.getenv("POSTGRES_CONNECTION_STRING")
        
        if pg_conn:
            # Parse connection string to set environment variables required by PG Storage
            # Format: postgresql://user:password@host:port/dbname
            regex = r"postgresql://(?P<user>[^:]+):(?P<password>[^@]+)@(?P<host>[^:/]+)(:(?P<port>\d+))?/(?P<db>.+)"
            match = re.match(regex, pg_conn)
            if match:
                os.environ["POSTGRES_USER"] = match.group("user")
                os.environ["POSTGRES_PASSWORD"] = match.group("password")
                os.environ["POSTGRES_HOST"] = match.group("host")
                os.environ["POSTGRES_PORT"] = match.group("port") or "5432"
                os.environ["POSTGRES_DATABASE"] = match.group("db")
                os.environ["POSTGRES_DB"] = match.group("db")

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

        # Correctly wrap the embedding function using LightRAG's internal dataclass
        # This prevents the TypeError with dataclasses.replace
        wrapped_emb = EmbeddingFunc(
            func=emb_func,
            embedding_dim=384,
            max_token_size=8192
        )

        try:
            if pg_conn:
                self.rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=llm_func, # LLM func can usually be direct
                    embedding_func=wrapped_emb,
                    kv_storage="PGKVStorage",
                    doc_status_storage="PGDocStatusStorage",
                    graph_storage="PGGraphStorage",
                    vector_storage="PGVectorStorage"
                )
            else:
                self.rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=llm_func,
                    embedding_func=wrapped_emb
                )
        except Exception as e:
            st.warning(f"Database storage failed, falling back to local: {e}")
            self.rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=llm_func,
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
