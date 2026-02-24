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
        self._initialized = False
        
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

        # Wrap the embedding function
        wrapped_emb = EmbeddingFunc(
            func=emb_func,
            embedding_dim=384,
            max_token_size=8192
        )

        # Initialize LightRAG (Internal local-first logic for stability)
        try:
            self.rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=llm_func,
                embedding_func=wrapped_emb
            )
        except Exception as e:
            st.error(f"Critical System Error: {e}")

    async def _ensure_initialized(self):
        """Ensures that LightRAG storages are initialized before any operation."""
        if not self._initialized:
            try:
                # Some versions use ainitialize_storages, some use initialize_storages
                if hasattr(self.rag, "ainitialize_storages"):
                    await self.rag.ainitialize_storages()
                elif hasattr(self.rag, "initialize_storages"):
                    await self.rag.initialize_storages()
                self._initialized = True
            except Exception as e:
                print(f"Initialization warning: {e}")

    def insert_text(self, text):
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run initialization and insertion
            async def run_insertion():
                await self._ensure_initialized()
                return await self.rag.ainsert(text)
                
            return loop.run_until_complete(run_insertion())
        except Exception as e:
            st.error(f"Error indexing text: {e}")
            return None

    def query(self, query, mode="hybrid"):
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def run_query():
                await self._ensure_initialized()
                return await self.rag.aquery(query, QueryParam(mode=mode))
                
            return loop.run_until_complete(run_query())
        except Exception as e:
            return f"Error querying: {e}"
