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
        
        self.rag = None
        self._loop = None
        self._initialized = False
        
        # Cache heavy models in session to make re-instantiation fast
        if "emb_model" not in st.session_state:
            from sentence_transformers import SentenceTransformer
            st.session_state.emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if "generator_instance" not in st.session_state:
            from handbook_generator import HandbookGenerator
            st.session_state.generator_instance = HandbookGenerator()
            
    def _create_rag_instance(self):
        """Creates the LightRAG instance with current context model."""
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

        return LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_func,
            embedding_func=wrapped_emb
        )

    async def _ensure_initialized(self):
        """Ensures that LightRAG is ready on the CURRENT loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(current_loop)

        # Only recreate if absolutely necessary (loop change or first time)
        if self.rag is None or self._loop != current_loop:
            self.rag = self._create_rag_instance()
            self._loop = current_loop
            self._initialized = False 

        if not self._initialized:
            try:
                if hasattr(self.rag, "initialize_storages"):
                    await self.rag.initialize_storages()
                elif hasattr(self.rag, "ainitialize_storages"):
                    await self.rag.ainitialize_storages()
                self._initialized = True
            except Exception as e:
                print(f"Init warning: {e}")

    def _get_loop(self):
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    async def ainsert_text(self, text):
        """Async version of text insertion with manual flush."""
        if not text: return
        await self._ensure_initialized()
        result = await self.rag.ainsert(text)
        return result

    def insert_text(self, text):
        """Sync wrapper for text insertion."""
        if not text: return False
        try:
            loop = self._get_loop()
            loop.run_until_complete(self.ainsert_text(text))
            return True
        except Exception as e:
            st.error(f"Error indexing text: {e}")
            return False

    async def aquery(self, query, mode="hybrid"):
        """Async version of querying."""
        await self._ensure_initialized()
        response = await self.rag.aquery(query, QueryParam(mode=mode))
        return response if response else "No relevant information found in the documents."

    def query(self, query, mode="hybrid"):
        """Sync wrapper for querying."""
        try:
            loop = self._get_loop()
            result = loop.run_until_complete(self.aquery(query, mode=mode))
            return result
        except Exception as e:
            st.error(f"Query Error: {e}")
            return f"I encountered an error querying the knowledge base."

def get_kb():
    """Singleton getter for KnowledgeBase to prevent session issues."""
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    return st.session_state.kb
