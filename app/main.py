import streamlit as st
import os
import asyncio
import sys
try:
    import pkg_resources
except ImportError:
    pass

from pdf_processor import extract_text_from_pdf
from knowledge_base import KnowledgeBase
from handbook_generator import HandbookGenerator
from dotenv import load_dotenv

load_dotenv()

# Page Config
st.set_page_config(page_title="AI Handbook Architect", layout="wide")

# Theme & Custom CSS (Vibrant/Premium)
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebarNav"]::before {
        content: "Drafting Suite";
        margin-left: 20px;
        margin-top: 20px;
        font-size: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

from knowledge_base import get_kb
from handbook_generator import get_generator

# Initialize Session State using safe getters
st.session_state.kb = get_kb()
st.session_state.generator = get_generator()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# Sidebar
with st.sidebar:
    st.title("Handbook Suite")
    uploaded_files = st.file_uploader("Upload Knowledge Source (PDF)", type="pdf", accept_multiple_files=True)
    
    if st.button("Index Documents"):
        if uploaded_files:
                total_chars = 0
                for uploaded_file in uploaded_files:
                    text = extract_text_from_pdf(uploaded_file)
                    if not text or len(text.strip()) < 10:
                        st.error(f"Could not read text from {uploaded_file.name}. Is it scanned or empty?")
                        continue
                    st.session_state.kb.insert_text(text)
                    total_chars += len(text)
                
                if total_chars > 0:
                    st.session_state.indexed = True
                    st.success(f"Indexing Complete! Successfully processed {total_chars:,} characters. Knowledge Graph is now active.")
                else:
                    st.error("Indexing failed: No readable content was found in the uploaded documents.")
        else:
            st.warning("Please upload at least one PDF.")
    
    if st.session_state.indexed:
        st.info("✅ System Ready: Knowledge Graph Active")
    else:
        st.warning("⚠️ Action Required: Please index your documents to enable Chat/Generation.")

# Main Interface
st.title("AI Handbook Architect")

if not st.session_state.indexed:
    st.info("### 👋 Welcome! \nTo begin, please **Upload a PDF** and click **'Index Documents'** in the sidebar. \n\n*Why? This app uses 'GraphRAG', which builds a complex web of relationships between ideas in your document. This is what allows for extremely high-quality 20,000-word handbooks, rather than just simple summaries.*")
else:
    st.write("Generate professional, long-form handbooks or chat with your document's Knowledge Graph.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question or request a handbook generation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                is_handbook_request = any(word in prompt.lower() for word in ["handbook", "guide", "manual", "book"])
                is_action_request = any(word in prompt.lower() for word in ["give me", "create", "generate", "write", "make", "tell me"])
                
                if is_handbook_request and is_action_request:
                    st.write("🚀 **Starting Long-Form Generation Engine...**")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(current, total, status):
                        progress_bar.progress(current / total)
                        status_text.text(f"Processing Section {current}/{total}: {status}")

                    handbook = st.session_state.generator.generate_handbook_sync(prompt, st.session_state.kb, update_progress)
                    
                    st.markdown(handbook)
                    st.download_button("📥 Download Generated Handbook", handbook, "generated_handbook.md")
                    st.session_state.messages.append({"role": "assistant", "content": "The handbook has been successfully generated. You can download the full version above."})
                else:
                    response = st.session_state.kb.query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
