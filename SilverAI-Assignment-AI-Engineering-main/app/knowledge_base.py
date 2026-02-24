import os
import asyncio
import streamlit as st
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
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
        
        # Create the wrapped functions exactly how LightRAG expects them
        embedding_function = EmbeddingFunc(
            embedding_dim=384, # Constant for all-MiniLM-L6-v2
            max_token_size=512,
            func=self._embedding_func
        )

        try:
            if pg_conn:
                self.rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=self._llm_complete,
                    embedding_func=embedding_function,
                    kv_storage="PGKVStorage",
                    doc_status_storage="PGDocStatusStorage",
                    graph_storage="PGGraphStorage",
                    vector_storage="PGVectorStorage",
                    addon_conf={"postgres_conn_config": pg_conn}
                )
            else:
                self.rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=self._llm_complete,
                    embedding_func=embedding_function
                )
        except Exception as e:
            # Final fallback
            self.rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=self._llm_complete,
                embedding_func=embedding_function
            )

    def _llm_complete(self, prompt, **kwargs):
        from handbook_generator import HandbookGenerator
        return HandbookGenerator()._get_completion(prompt)

    async def _embedding_func(self, texts):
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
