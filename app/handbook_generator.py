import os
import asyncio
import time
import re
import streamlit as st
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class HandbookGenerator:
    def __init__(self, model_provider=None, model_name=None):
        # Determine model provider and name from secrets or env
        self.model_provider = model_provider or st.secrets.get("MODEL_PROVIDER") or os.getenv("MODEL_PROVIDER", "grok")
        self.model_name = model_name or st.secrets.get("MODEL_NAME") or os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        self.fallback_model = "llama-3.1-8b-instant"
        
        # Get API Key from Streamlit Secrets or Environment
        api_key = st.secrets.get("GROK_API_KEY") or os.getenv("GROK_API_KEY")
        
        if self.model_provider == "grok":
            if not api_key:
                raise ValueError("GROK_API_KEY not found in Streamlit Secrets or .env file")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            self.async_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        else:
            openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=openai_key)
            self.async_client = AsyncOpenAI(api_key=openai_key)

    def _get_completion(self, prompt, max_tokens=4096, retry_count=3, current_model_override=None):
        for attempt in range(retry_count):
            current_model = current_model_override or self.model_name
            try:
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                err_msg = str(e)
                if "rate_limit_exceeded" in err_msg or "429" in err_msg:
                    if not current_model_override and self.model_name != self.fallback_model:
                        self.model_name = self.fallback_model
                        continue
                    wait_time = 60
                    match = re.search(r"try again in ([\d\.]+)s", err_msg)
                    if match: wait_time = float(match.group(1)) + 2
                    time.sleep(wait_time)
                elif "decommissioned" in err_msg or "400" in err_msg:
                    if not current_model_override: self.model_name = self.fallback_model
                    continue
                else:
                    time.sleep(5)
        return "I encountered a persistent error with the AI provider."

    async def _get_completion_async(self, prompt, max_tokens=4096, retry_count=3, current_model_override=None):
        for attempt in range(retry_count):
            current_model = current_model_override or self.model_name
            try:
                response = await self.async_client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                err_msg = str(e)
                if "rate_limit_exceeded" in err_msg or "429" in err_msg:
                    if not current_model_override and self.model_name != self.fallback_model:
                        self.model_name = self.fallback_model
                        continue
                    wait_time = 60
                    match = re.search(r"try again in ([\d\.]+)s", err_msg)
                    if match: wait_time = float(match.group(1)) + 2
                    await asyncio.sleep(wait_time)
                elif "decommissioned" in err_msg or "400" in err_msg:
                    if not current_model_override: self.model_name = self.fallback_model
                    continue
                else:
                    await asyncio.sleep(5)
        return "I encountered a persistent error with the AI provider."

    def generate_handbook_sync(self, prompt, kb, progress_callback=None):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.generate_handbook(prompt, kb, progress_callback))

    async def generate_handbook(self, prompt, kb, progress_callback=None):
        # 1. Faster context retrieval ('naive' is 10x faster than 'local' for large docs)
        if progress_callback: progress_callback(0, 20, "Searching Knowledge Base (Fast Mode)...")
        context = await kb.aquery(prompt, mode="naive") 
        
        # 2. Plan Generation (Keep 70B for high-quality structure)
        if progress_callback: progress_callback(0, 20, "Architecting Handbook Structure...")
        plan_prompt = f"Break down this request into 20 distinct chapter titles: {prompt}. Context: {context[:3000]}"
        plan_text = await self._get_completion_async(plan_prompt) # Uses 70B by default
        
        sections = [line.strip() for line in plan_text.split("\n") if line.strip() and ("Section" in line or any(c.isdigit() for c in line[:3]))]
        if not sections: sections = [f"Section {i}: Part {i}" for i in range(1, 21)]
        sections = sections[:20]

        # 3. Parallel Execution with 8B Model for Speed
        # Use 8B model for drafting to avoid rate limits and generate content 5x faster
        generation_model = self.fallback_model # llama-3.1-8b-instant
        semaphore = asyncio.Semaphore(5)
        completed_count = 0

        async def generate_section(i, section_step):
            nonlocal completed_count
            async with semaphore:
                if progress_callback: progress_callback(completed_count, len(sections), f"Drafting: {section_step}")
                write_prompt = f"""As a professional author, write a detailed, 1000-word chapter for: {section_step}. 
                Handbook Topic: {prompt}
                Context snippets: {context[:2000]}
                Style: Deep, educational, and professional. 
                Length: Maximum possible detail."""
                
                # Force the use of the faster 8B model for chapter drafting
                content = await self._get_completion_async(write_prompt, current_model_override=generation_model)
                completed_count += 1
                if progress_callback: progress_callback(completed_count, len(sections), f"Done: {section_step}")
                return content

        # Run sections in parallel
        tasks = [generate_section(i, s) for i, s in enumerate(sections)]
        results = await asyncio.gather(*tasks)
        
        return "\n\n".join(results)

def get_generator():
    """Singleton getter for HandbookGenerator."""
    if "generator_instance" not in st.session_state:
        st.session_state.generator_instance = HandbookGenerator()
    return st.session_state.generator_instance
