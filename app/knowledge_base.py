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
        
        # Cache heavy models in session to make re-instantiation fast
        if "emb_model" not in st.session_state:
            from sentence_transformers import SentenceTransformer
            st.session_state.emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if "generator_instance" not in st.session_state:
            from handbook_generator import HandbookGenerator
            st.session_state.generator_instance = HandbookGenerator()

    async def _get_active_rag(self):
        """Creates and returns a fresh RAG instance bound to the CURRENT loop."""
        def llm_func(prompt, **kwargs):
            gen = st.session_state.generator_instance
            return gen._get_completion(prompt, current_model_override=gen.fallback_model)

        async def emb_func(texts):
            if isinstance(texts, str):
                texts = [texts]
            return st.session_state.emb_model.encode(texts)

        wrapped_emb = EmbeddingFunc(
            func=emb_func,
            embedding_dim=384,
            max_token_size=8192
        )

        # Create fresh instance for this operation ONLY
        rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_func,
            embedding_func=wrapped_emb
        )

        # Initialize its storages on the current loop
        if hasattr(rag, "initialize_storages"):
            await rag.initialize_storages()
        elif hasattr(rag, "ainitialize_storages"):
            await rag.ainitialize_storages()
            
        return rag

    def _get_loop(self):
        """Helper to safely get or create an event loop for sync wrappers."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    async def ainsert_text(self, text):
        """Async version of text insertion."""
        if not text: return
        rag = await self._get_active_rag()
        return await rag.ainsert(text)

    def insert_text(self, text):
        """Sync wrapper for text insertion."""
        if not text: return False
        try:
            loop = self._get_loop()
            result = loop.run_until_complete(self.ainsert_text(text))
            return result is not None
        except Exception as e:
            st.error(f"Error indexing text: {e}")
            return False

    async def aquery(self, query, mode="hybrid"):
        """Async version of querying."""
        rag = await self._get_active_rag()
        return await rag.aquery(query, QueryParam(mode=mode))

    def query(self, query, mode="hybrid"):
        """Sync wrapper for querying."""
        try:
            loop = self._get_loop()
            return loop.run_until_complete(self.aquery(query, mode=mode))
        except Exception as e:
            st.error(f"Query Error: {e}")
            return f"I encountered an error querying the knowledge base."

def get_kb():
    """Singleton getter for KnowledgeBase to prevent session issues."""
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    return st.session_state.kb
