import os
import asyncio
import time
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class HandbookGenerator:
    def __init__(self, model_provider="grok", model_name="llama-3.3-70b-versatile"):
        self.model_provider = model_provider
        self.model_name = model_name
        self.fallback_model = "llama-3.1-8b-instant" # UPDATED: Using the current instant model
        
        if model_provider == "grok":
            self.client = OpenAI(
                api_key=os.getenv("GROK_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
                # Handle Rate Limits (429)
                if "rate_limit_exceeded" in err_msg or "429" in err_msg:
                    if self.model_name != self.fallback_model:
                        print(f"WARNING: 70B Rate limit hit. Swapping to {self.fallback_model} for speed...")
                        self.model_name = self.fallback_model
                        continue # Re-try immediately with the 8B model
                    
                    wait_time = 60
                    match = re.search(r"try again in ([\d\.]+)s", err_msg)
                    if match:
                        wait_time = float(match.group(1)) + 2
                    
                    print(f"WAIT: Global Rate limit hit. Sleeping for {wait_time}s...")
                    time.sleep(wait_time)
                
                # Handle Decommissioned/Wrong Model Errors (400)
                elif "decommissioned" in err_msg or "400" in err_msg:
                    print(f"FIX: Model {current_model} unavailable. Forcing switch to {self.fallback_model}...")
                    self.model_name = self.fallback_model
                    continue
                
                else:
                    print(f"Error: {e}")
                    time.sleep(5) # Small cooldown for unknown errors
                    
        return "I encountered a persistent error with the AI provider. Please try again in a few minutes."

    def generate_handbook_sync(self, prompt, kb, progress_callback=None):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.generate_handbook(prompt, kb, progress_callback))

    async def generate_handbook(self, prompt, kb, progress_callback=None):
        print("DEBUG: Fetching context for handbook...")
        context = kb.query(prompt, mode="hybrid")
        
        # Phase 1: Planning
        plan_prompt = f"""Break down this request into 20 detailed sections for a 20,000-word handbook.
Prompt: {prompt}
Context: {context[:2000]}
Format: Section X - Title: [Title] - Words: 1000"""
        
        plan_text = self._get_completion(plan_prompt)
        sections = [line for line in plan_text.split("\n") if "Section" in line]
        
        if not sections:
             sections = [f"Section {i}: Deep Dive Part {i} - Words: 1000" for i in range(1, 21)]

        full_content = []
        already_written = ""
        
        # Phase 2: Sequential Writing
        for i, section_step in enumerate(sections):
            if progress_callback:
                progress_callback(i + 1, len(sections), f"Writing {section_step}...")
            
            write_prompt = f"""Write a massive 1000-word section for: {section_step}. 
Use this context: {context[:1500]}
Current Handbook progress: {already_written[-1000:]}
Output ONLY the section content."""
            
            section_text = self._get_completion(write_prompt)
            full_content.append(section_text)
            already_written += section_text
            
        return "\n\n".join(full_content)
