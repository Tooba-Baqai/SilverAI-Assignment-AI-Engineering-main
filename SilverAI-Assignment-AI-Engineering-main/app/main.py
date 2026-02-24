import streamlit as st
import os
import asyncio
from pdf_processor import extract_text_from_pdf
from knowledge_base import KnowledgeBase
from handbook_generator import HandbookGenerator
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LunarTech Handbook Generator", layout="wide")

# Initialize Session State
if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()
if "generator" not in st.session_state:
    st.session_state.generator = HandbookGenerator(
        model_provider=os.getenv("MODEL_PROVIDER", "grok"),
        model_name=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    )
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("📚 LunarTech AI")
    st.subheader("Handbook Generator")
    
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                for uploaded_file in uploaded_files:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    text = extract_text_from_pdf(temp_path)
                    st.session_state.kb.insert_text(text)
                    os.remove(temp_path)
                st.success(f"Processed {len(uploaded_files)} documents!")
        else:
            st.warning("Please upload at least one PDF.")

    st.divider()
    st.write("Settings")
    st.write(f"**LLM:** {st.session_state.generator.model_name}")

# Main Interface
st.header("Chat with your Knowledge Graph")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question or request a handbook..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            is_handbook_request = any(word in prompt.lower() for word in ["handbook", "guide", "manual", "book"])
            is_action_request = any(word in prompt.lower() for word in ["give me", "create", "generate", "write", "make", "tell me"])
            
            if is_handbook_request and is_action_request:
                st.write("🚀 **Starting LongWriter Engine (20k word target)...**")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, status):
                    progress_bar.progress(current / total)
                    status_text.text(f"Section {current}/{total}: {status}")

                handbook = st.session_state.generator.generate_handbook_sync(prompt, st.session_state.kb, update_progress)
                
                st.markdown(handbook)
                st.download_button("📥 Download Full Handbook", handbook, "lunartech_handbook.md")
                st.session_state.messages.append({"role": "assistant", "content": "I have finished generating your comprehensive handbook! You can download it above."})
            else:
                response = st.session_state.kb.query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
