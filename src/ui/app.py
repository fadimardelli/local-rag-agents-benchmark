import streamlit as st
import time
from pathlib import Path

# --- Import Fadi's Backend Code ---
from src.rag.traditional import TraditionalRAG
from src.config.defaults import LLAMA_GGUF_PATH_8B

# --- Page Configuration ---
st.set_page_config(
    page_title="LexLocal | Thesis UI",
    page_icon="⚖️",
    layout="wide"
)

# --- 1. THE HARDWARE TRAP FIX (Load Models Once) ---
@st.cache_resource(show_spinner="Loading Llama-3 and FAISS into Memory... Please wait.")
def initialize_backend():
    """Initializes the heavy models only once."""
    # We use the 8B model default path. 
    # Fadi's TraditionalRAG automatically handles the FAISS index and Llama setup.
    rag_engine = TraditionalRAG(model_path=LLAMA_GGUF_PATH_8B)
    return rag_engine

# Call the initialization
rag = initialize_backend()

# --- Initialize Chat History & Telemetry in Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0

# --- Sidebar: Controls & Telemetry ---
with st.sidebar:
    st.title("LexLocal")
    st.subheader("Resource-Aware Legal Assistant")
    st.divider()
    
    st.markdown("**1. Workflow Strategy**")
    workflow = st.radio(
        "Select Pipeline:",
        ["⚡ Traditional RAG (System A)", "🧠 Agentic Loop (System B)"],
        help="System A is a linear single-shot search. System B uses an iterative reflection loop."
    )
    
    st.divider()
    
    st.markdown("**3. Hardware Telemetry**")
    st.caption("Live Local Metrics (Llama-3-8B)")
    
    # We update the latency dynamically based on the last run
    st.metric(label="Last Query Latency", value=f"{st.session_state.last_latency:.2f} s")
    st.metric(label="VRAM Config", value="Q4_K_M (Quantized)")

# --- Main Chat Interface ---
st.header("Legal Document Q&A")

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If the message has retrieved evidence attached, display it in an expander
        if "evidence" in msg and msg["evidence"]:
            with st.expander("🔍 View Retrieved Evidence"):
                for chunk in msg["evidence"]:
                    st.info(chunk)

# User Input
if prompt := st.chat_input("Ask a question about the contract..."):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response Logic
    with st.chat_message("assistant"):
        
        # --- IF TRADITIONAL RAG (SYSTEM A) ---
        if "Traditional" in workflow:
            with st.spinner("System A: Retrieving and generating..."):
                
                # Start the timer!
                start_time = time.time()
                
                # RUN FADI'S CODE
                result = rag.run(prompt)
                
                # Stop the timer!
                end_time = time.time()
                st.session_state.last_latency = end_time - start_time
                
                # Display the final answer
                st.write(result.answer)
                
                # Format the evidence chunks for the UI
                evidence_list = []
                with st.expander("🔍 View Retrieved Evidence", expanded=False):
                    for i, chunk in enumerate(result.used, 1):
                        evidence_text = f"**Source:** {chunk.doc_path} (Chars: {chunk.start}-{chunk.end}) | **Relevance Score:** {chunk.score:.4f}\n\n> {chunk.text.strip()}"
                        evidence_list.append(evidence_text)
                        st.info(evidence_text)
                        
                # Save assistant message and evidence to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result.answer,
                    "evidence": evidence_list
                })
                
                # Force a UI refresh to update the telemetry sidebar
                st.rerun()
                
        # --- IF AGENTIC RAG (SYSTEM B) ---
        else:
            st.warning("Agentic RAG is not hooked up yet! Switch back to Traditional RAG.")