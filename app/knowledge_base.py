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
            
    def _create_fresh_rag(self):
        """Creates a local, loop-bound RAG instance."""
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

    def _run_factory(self, func, *args, **kwargs):
        """The 'Clean Room' Factory: Creates a fresh loop for every task."""
        async def _internal():
            rag = self._create_fresh_rag()
            # Explicitly wait for storage sync before answering
            if hasattr(rag, "initialize_storages"):
                await rag.initialize_storages()
            elif hasattr(rag, "ainitialize_storages"):
                await rag.ainitialize_storages()
            
            return await func(rag, *args, **kwargs)

        try:
            # Create a private world (loop) for this operation
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_internal())
            finally:
                new_loop.close()
        except Exception as e:
            st.error(f"AI Engine Error: {e}")
            return None

    def insert_text(self, text):
        """Sync indexing in an isolated loop."""
        if not text: return False
        
        async def _insert(rag, t):
            await rag.ainsert(t)
            return True

        result = self._run_factory(_insert, text)
        print(f"Indexing outcome: {'Success' if result else 'Failed'}")
        return bool(result)

    def query(self, query, mode="hybrid"):
        """Sync query in an isolated loop with failure fallback."""
        
        async def _query(rag, q, m):
            def is_bad(t):
                if not t: return True
                low = str(t).lower()
                failures = ["sorry", "i'm not able", "no-context", "i couldn't find", "not able to provide"]
                return len(str(t).strip()) < 15 or any(f in low for f in failures)

            # Try primary mode
            res = await rag.aquery(q, QueryParam(mode=m))
            
            # Fallback cascade to ensure we never get "None"
            if is_bad(res):
                res = await rag.aquery(q, QueryParam(mode="local"))
            if is_bad(res):
                res = await rag.aquery(q, QueryParam(mode="naive"))
                
            return res if not is_bad(res) else "I scanned the documents but couldn't find a direct answer. Please ensure the files were indexed correctly."

        return self._run_factory(_query, query, mode)

    async def aquery(self, query, mode="hybrid"):
        """Async version for the generator engine."""
        # Generator uses its own loop management, so we just provide the instance
        rag = self._create_fresh_rag()
        if hasattr(rag, "initialize_storages"):
            await rag.initialize_storages()
        return await rag.aquery(query, QueryParam(mode=mode))

def get_kb():
    """Singleton getter for KnowledgeBase to prevent session issues."""
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    return st.session_state.kb
