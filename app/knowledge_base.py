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
        
        # Cache heavy models in session to keep 'Isolated Factory' fast
        if "emb_model" not in st.session_state:
            from sentence_transformers import SentenceTransformer
            st.session_state.emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if "generator_instance" not in st.session_state:
            from handbook_generator import HandbookGenerator
            st.session_state.generator_instance = HandbookGenerator()
            
    def _create_isolated_rag(self):
        """Standard RAG constructor used inside the isolated loop."""
        def llm_func(prompt, **kwargs):
            return st.session_state.generator_instance._get_completion(
                prompt, current_model_override=st.session_state.generator_instance.fallback_model
            )

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

    def insert_text(self, text):
        """Isolated Loop Factory for insertion."""
        if not text: return False
        
        async def _run():
            rag = self._create_isolated_rag()
            if hasattr(rag, "initialize_storages"):
                await rag.initialize_storages()
            elif hasattr(rag, "ainitialize_storages"):
                await rag.ainitialize_storages()
            await rag.ainsert(text)
            return True

        try:
            # Create a completely fresh loop for this specific task
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(_run())
        except Exception as e:
            st.error(f"Indexing Error: {e}")
            return False

    def query(self, query, mode="hybrid"):
        """Isolated Loop Factory for querying."""
        
        async def _run():
            rag = self._create_isolated_rag()
            if hasattr(rag, "initialize_storages"):
                await rag.initialize_storages()
            elif hasattr(rag, "ainitialize_storages"):
                await rag.ainitialize_storages()
            
            response = await rag.aquery(query, QueryParam(mode=mode))
            return response if response else "I couldn't find an answer in the document."

        try:
            # Create a completely fresh loop for this specific task
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result = new_loop.run_until_complete(_run())
            return result
        except Exception as e:
            st.error(f"Query Error: {e}")
            return "I encountered a technical error while searching the document."

    async def aquery(self, query, mode="hybrid"):
        """Async compatibility for the generator engine."""
        # Use simple aquery for the generator which handles its own loops
        rag = self._create_isolated_rag()
        if hasattr(rag, "initialize_storages"):
            await rag.initialize_storages()
        return await rag.aquery(query, QueryParam(mode=mode))

def get_kb():
    """Singleton getter for KnowledgeBase to prevent session issues."""
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    return st.session_state.kb
