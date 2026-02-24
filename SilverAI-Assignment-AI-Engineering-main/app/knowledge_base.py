import os
import asyncio
import threading
import numpy as np
from lightrag import LightRAG, QueryParam
from dotenv import load_dotenv

load_dotenv()

class KnowledgeBase:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Handle Singleton pattern to avoid re-initializing background thread."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(KnowledgeBase, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, working_dir="./lightrag_storage"):
        if self._initialized:
            return
        
        self.working_dir = working_dir
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
            
        self.pg_conn = os.getenv("POSTGRES_CONNECTION_STRING")
        self.rag = None
        self._model = None
        
        # Start the background event loop thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_background_loop, daemon=True)
        self.thread.start()
        
        self._initialized = True
        print("DEBUG: AI Engine started on dedicated background thread.")

    def _run_background_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _init_rag_instance(self):
        """Internal helper to create a fresh RAG instance inside the background loop."""
        async def llm_model_func(prompt, system_prompt=None, history=None, **kwargs) -> str:
            from handbook_generator import HandbookGenerator
            gen = HandbookGenerator(model_provider="grok")
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            return gen._get_completion(full_prompt)

        async def embedding_func(texts: list[str]) -> np.ndarray:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and "YOUR_OPENAI_KEY" not in openai_key and openai_key.strip() != "":
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=openai_key)
                response = await client.embeddings.create(model="text-embedding-3-small", input=texts)
                return np.array([item.embedding for item in response.data])
            else:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer('all-MiniLM-L6-v2')
                return self._model.encode(texts)

        from lightrag.utils import EmbeddingFunc
        openai_key = os.getenv("OPENAI_API_KEY")
        model_name = "text-embedding-3-small" if (openai_key and "YOUR_OPENAI_KEY" not in openai_key) else "all-MiniLM-L6-v2"
        dim = 1536 if model_name == "text-embedding-3-small" else 384
        
        emb_func = EmbeddingFunc(embedding_dim=dim, max_token_size=8192, func=embedding_func, model_name=model_name)
        addon_params = {"entity_types": ["organization", "person", "technology", "event", "concept"]}

        if self.pg_conn:
            os.environ["POSTGRES_STATEMENT_CACHE_SIZE"] = "0"
            return LightRAG(
                working_dir=self.working_dir,
                llm_model_func=llm_model_func,
                embedding_func=emb_func,
                kv_storage="PGKVStorage",
                doc_status_storage="PGDocStatusStorage",
                graph_storage="PGGraphStorage",
                vector_storage="PGVectorStorage",
                addon_params=addon_params
            )
        else:
            return LightRAG(working_dir=self.working_dir, llm_model_func=llm_model_func, embedding_func=emb_func, addon_params=addon_params)

    async def _async_setup(self):
        """Must run inside the background loop."""
        if self.rag is None:
            self.rag = self._init_rag_instance()
            try:
                await self.rag.initialize_storages()
            except Exception as e:
                print(f"DEBUG: PG Init failed ({e}), Falling back to local.")
                self.pg_conn = None
                self.rag = self._init_rag_instance()
                await self.rag.initialize_storages()

    def run_sync(self, coro):
        """Helper to run async code safely from the Streamlit UI thread."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def insert_text(self, text):
        def _task():
            async def _do():
                await self._async_setup()
                print(f"DEBUG: Processing document ({len(text)} characters)...")
                await self.rag.ainsert(text)
                print("DEBUG: Success!")
            return _do()
        return self.run_sync(_task())

    def query(self, query_text, mode="hybrid"):
        def _task():
            async def _do():
                await self._async_setup()
                print(f"DEBUG: Querying {mode}...")
                return await self.rag.aquery(query_text, param=QueryParam(mode=mode))
            return _do()
        return self.run_sync(_task())
