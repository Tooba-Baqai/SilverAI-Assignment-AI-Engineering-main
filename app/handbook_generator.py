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

    def _get_completion(self, prompt, max_tokens=4096, retry_count=3):
        for attempt in range(retry_count):
            current_model = self.model_name
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
                    if self.model_name != self.fallback_model:
                        self.model_name = self.fallback_model
                        continue
                    wait_time = 60
                    match = re.search(r"try again in ([\d\.]+)s", err_msg)
                    if match: wait_time = float(match.group(1)) + 2
                    time.sleep(wait_time)
                elif "decommissioned" in err_msg or "400" in err_msg:
                    self.model_name = self.fallback_model
                    continue
                else:
                    time.sleep(5)
        return "I encountered a persistent error with the AI provider."

    async def _get_completion_async(self, prompt, max_tokens=4096, retry_count=3):
        for attempt in range(retry_count):
            current_model = self.model_name
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
                    if self.model_name != self.fallback_model:
                        self.model_name = self.fallback_model
                        continue
                    wait_time = 60
                    match = re.search(r"try again in ([\d\.]+)s", err_msg)
                    if match: wait_time = float(match.group(1)) + 2
                    await asyncio.sleep(wait_time)
                elif "decommissioned" in err_msg or "400" in err_msg:
                    self.model_name = self.fallback_model
                    continue
                else:
                    await asyncio.sleep(5)
        return "I encountered a persistent error with the AI provider."

    def generate_handbook_sync(self, prompt, kb, progress_callback=None):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.generate_handbook(prompt, kb, progress_callback))

    async def generate_handbook(self, prompt, kb, progress_callback=None):
        # 1. Faster context retrieval (local mode is much faster than hybrid for large docs)
        if progress_callback: progress_callback(0, 20, "Scanning Knowledge Graph...")
        context = await kb.aquery(prompt, mode="local") 
        
        # 2. Plan Generation
        if progress_callback: progress_callback(0, 20, "Architecting Handbook Structure...")
        plan_prompt = f"Break down this request into 20 distinct, detailed chapter titles: {prompt}. Context: {context[:3000]}"
        plan_text = await self._get_completion_async(plan_prompt)
        
        sections = [line.strip() for line in plan_text.split("\n") if line.strip() and ("Section" in line or any(c.isdigit() for c in line[:3]))]
        if not sections: sections = [f"Section {i}: Part {i}" for i in range(1, 21)]
        sections = sections[:20]

        # 3. Parallel Execution with Concurrency Limit (Semaphore)
        # Using a limit of 4 to balance speed and Groq rate limits (14.4k tokens per minute)
        semaphore = asyncio.Semaphore(4)
        completed_count = 0

        async def generate_section(i, section_step):
            nonlocal completed_count
            async with semaphore:
                if progress_callback: progress_callback(completed_count, len(sections), f"Drafting: {section_step}")
                write_prompt = f"""Write a comprehensive chapter for: {section_step}. 
                Handbook Topic: {prompt}
                Context: {context[:1500]}
                Target: ~1000 words. Maintain professional depth."""
                
                content = await self._get_completion_async(write_prompt)
                completed_count += 1
                if progress_callback: progress_callback(completed_count, len(sections), f"Completed: {section_step}")
                return content

        # Run sections in parallel
        tasks = [generate_section(i, s) for i, s in enumerate(sections)]
        results = await asyncio.gather(*tasks)
        
        return "\n\n".join(results)
