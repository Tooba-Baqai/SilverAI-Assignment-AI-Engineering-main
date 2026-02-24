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
        
        self._initialized = False
        
        # Use session state to cache the heavy model so it's only loaded ONCE
        if "emb_model" not in st.session_state:
            from sentence_transformers import SentenceTransformer
            st.session_state.emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if "generator_instance" not in st.session_state:
            from handbook_generator import HandbookGenerator
            st.session_state.generator_instance = HandbookGenerator()

        # Define the processing functions using the cached model
        def llm_func(prompt, **kwargs):
            # Internal RAG operations (extraction, summarizing) are better on 8B for stability
            gen = st.session_state.generator_instance
            return gen._get_completion(prompt, current_model_override=gen.fallback_model)

        async def emb_func(texts):
            if isinstance(texts, str):
                texts = [texts]
            return st.session_state.emb_model.encode(texts)

        # Wrap the embedding function
        wrapped_emb = EmbeddingFunc(
            func=emb_func,
            embedding_dim=384,
            max_token_size=8192
        )

        # Initialize LightRAG
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

    def _get_loop(self):
        """Helper to safely get or create an event loop."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    async def ainsert_text(self, text):
        """Async version of text insertion."""
        await self._ensure_initialized()
        return await self.rag.ainsert(text)

    def insert_text(self, text):
        """Sync wrapper for text insertion."""
        try:
            loop = self._get_loop()
            return loop.run_until_complete(self.ainsert_text(text))
        except Exception as e:
            # If still failing, try to fix the lock issue by re-initializing the internal lock if possible
            st.error(f"Error indexing text: {e}")
            return None

    async def aquery(self, query, mode="hybrid"):
        """Async version of querying."""
        await self._ensure_initialized()
        return await self.rag.aquery(query, QueryParam(mode=mode))

    def query(self, query, mode="hybrid"):
        """Sync wrapper for querying."""
        try:
            loop = self._get_loop()
            return loop.run_until_complete(self.aquery(query, mode=mode))
        except Exception as e:
            return f"Error querying: {e}"

def get_kb():
    """Singleton getter for KnowledgeBase to prevent event loop issues."""
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    return st.session_state.kb
