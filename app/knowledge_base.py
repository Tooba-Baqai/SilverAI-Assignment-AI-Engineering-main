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

import threading

class KnowledgeBase:
    def __init__(self, working_dir="./lightrag_storage"):
        self.working_dir = working_dir
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        
        # Models are heavy, keep them in session state
        if "emb_model" not in st.session_state:
            from sentence_transformers import SentenceTransformer
            st.session_state.emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if "generator_instance" not in st.session_state:
            from handbook_generator import HandbookGenerator
            st.session_state.generator_instance = HandbookGenerator()

    def _create_rag_core(self):
        """Standard RAG constructor."""
        def llm_func(prompt, **kwargs):
            return st.session_state.generator_instance._get_completion(
                prompt, current_model_override=st.session_state.generator_instance.fallback_model
            )

        async def emb_func(texts):
            if isinstance(texts, str):
                texts = [texts]
            return st.session_state.emb_model.encode(texts)

        return LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_func,
            embedding_func=EmbeddingFunc(
                func=emb_func,
                embedding_dim=384,
                max_token_size=8192
            )
        )

    def _threaded_worker(self, result_container, task_type, *args, **kwargs):
        """Worker function that runs in a dedicated thread with its own loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def _execute():
            rag = self._create_rag_core()
            # Absolute requirement for this version of LightRAG
            if hasattr(rag, "initialize_storages"):
                await rag.initialize_storages()
            elif hasattr(rag, "ainitialize_storages"):
                await rag.ainitialize_storages()
            
            if task_type == "insert":
                await rag.ainsert(args[0])
                return True
            elif task_type == "query":
                res = await rag.aquery(args[0], QueryParam(mode=kwargs.get("mode", "hybrid")))
                
                # Internal fallback logic to prevent None/No-Context
                def is_fail(t):
                    if not t or len(str(t)) < 15: return True
                    l = str(t).lower()
                    return any(f in l for f in ["sorry", "i'm not able", "no-context", "i couldn't find"])
                
                if is_fail(res):
                    res = await rag.aquery(args[0], QueryParam(mode="local"))
                if is_fail(res):
                    res = await rag.aquery(args[0], QueryParam(mode="naive"))
                
                return res if not is_fail(res) else "I scanned the documents but couldn't find a specific answer. Please try re-phrasing."

        try:
            result_container["result"] = loop.run_until_complete(_execute())
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            loop.close()

    def insert_text(self, text):
        """Thread-bridged insertion."""
        if not text: return False
        res_container = {"result": None, "error": None}
        thread = threading.Thread(target=self._threaded_worker, args=(res_container, "insert", text))
        thread.start()
        thread.join()
        
        if res_container["error"]:
            st.error(f"Indexing Engine Error: {res_container['error']}")
            return False
        return True

    def query(self, query, mode="hybrid"):
        """Thread-bridged query."""
        res_container = {"result": None, "error": None}
        thread = threading.Thread(target=self._threaded_worker, args=(res_container, "query", query), kwargs={"mode": mode})
        thread.start()
        thread.join()
        
        if res_container["error"]:
            return f"Technical Error: {res_container['error']}"
        return res_container["result"]

    async def aquery(self, query, mode="hybrid"):
        """Async version specifically for the Generator engine (which manages its own loop)."""
        rag = self._create_rag_core()
        if hasattr(rag, "initialize_storages"): await rag.initialize_storages()
        return await rag.aquery(query, QueryParam(mode=mode))

def get_kb():
    """Singleton getter for KnowledgeBase to prevent session issues."""
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    return st.session_state.kb
