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
        
        # Cache models and instances in session state for cross-script persistence
        if "emb_model" not in st.session_state:
            from sentence_transformers import SentenceTransformer
            st.session_state.emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if "generator_instance" not in st.session_state:
            from handbook_generator import HandbookGenerator
            st.session_state.generator_instance = HandbookGenerator()
            
        if "rag_instance" not in st.session_state:
            st.session_state.rag_instance = None
            st.session_state.rag_loop = None

    def _create_rag_core(self):
        """Creates the internal LightRAG object."""
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

    async def _ensure_rag(self):
        """Ensures the RAG instance is alive and on the CORRECT event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(current_loop)

        # Check if we need to (re)create the RAG instance
        if st.session_state.rag_instance is None or st.session_state.rag_loop != current_loop:
            st.session_state.rag_instance = self._create_rag_core()
            st.session_state.rag_loop = current_loop
            
            # Re-initialize storage on this new instance/loop
            if hasattr(st.session_state.rag_instance, "initialize_storages"):
                await st.session_state.rag_instance.initialize_storages()
            elif hasattr(st.session_state.rag_instance, "ainitialize_storages"):
                await st.session_state.rag_instance.ainitialize_storages()
        
        return st.session_state.rag_instance

    def insert_text(self, text):
        """Sync wrapper with loop safety and character count logging."""
        if not text or len(text.strip()) < 10: 
            print("Indexing warning: Provided text is too short or empty.")
            return False
            
        async def _internal():
            rag = await self._ensure_rag()
            await rag.ainsert(text)
            print(f"Successfully indexed {len(text)} characters into Knowledge Graph.")
            return True

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_internal())
        except Exception as e:
            st.error(f"Sync Indexing Error: {e}")
            return False

    def query(self, query, mode="hybrid"):
        """Adaptive Query with Fallback Search and failure detection."""
        
        async def _internal():
            rag = await self._ensure_rag()
            
            def is_failure(text):
                if not text: return True
                text_lower = str(text).lower()
                # LightRAG often returns these phrases when no context is found
                failures = ["sorry", "i'm not able", "no-context", "don't have enough information", "i couldn't find"]
                if len(text_lower.strip()) < 20: return True
                return any(f in text_lower for f in failures)

            # Try primary search mode
            res = await rag.aquery(query, QueryParam(mode=mode))
            
            # Fallback logic: If failure detected, try other modes
            if is_failure(res):
                res = await rag.aquery(query, QueryParam(mode="local"))
            
            if is_failure(res):
                res = await rag.aquery(query, QueryParam(mode="naive"))
                
            if is_failure(res):
                return "I've scanned the documents but couldn't find a direct answer. Please ensure the documents were indexed correctly or try re-phrasing your question."
                
            return res

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_internal())
        except Exception as e:
            st.error(f"Sync Query Error: {e}")
            return "The AI engine encountered an error. Please try clicking 'Index Documents' again."

    async def aquery(self, query, mode="hybrid"):
        """Direct async query for high-performance generation."""
        rag = await self._ensure_rag()
        return await rag.aquery(query, QueryParam(mode=mode))

def get_kb():
    """Singleton getter for KnowledgeBase to prevent session issues."""
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    return st.session_state.kb
