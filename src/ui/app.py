import streamlit as st
import time
from pathlib import Path

# --- Import Backend Components ---
from src.rag.traditional import TraditionalRAG
from src.rag.agentic import AgenticRAG
from src.config.defaults import LLAMA_GGUF_PATH_8B

# --- Page Configuration ---
st.set_page_config(
    page_title="LexLocal | Thesis Benchmarking",
    page_icon="⚖️",
    layout="wide"
)

# --- 1. SHARED RESOURCE INITIALIZATION ---
@st.cache_resource(show_spinner="Initializing Llama-3 and FAISS Index...")
def initialize_systems():
    """
    Loads models once. AgenticRAG reuses the model and retriever 
    from TraditionalRAG to save memory.
    """
    # 1. Initialize System A
    sys_a = TraditionalRAG(model_path=LLAMA_GGUF_PATH_8B)
    
    # 2. Initialize System B (reusing internal objects)
    sys_b = AgenticRAG(
        retriever=sys_a.retriever,
        model=sys_a.model
    )
    return sys_a, sys_b

# Get the engines
system_a, system_b = initialize_systems()

# --- 2. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0
if "last_iterations" not in st.session_state:
    st.session_state.last_iterations = 1

# --- 3. SIDEBAR TELEMETRY ---
with st.sidebar:
    st.title("LexLocal")
    st.subheader("Hardware-Aware Research")
    st.divider()
    
    workflow = st.radio(
        "Select Research Pipeline:",
        ["⚡ Traditional RAG (System A)", "🧠 Agentic Loop (System B)"],
        help="System A: Single-shot retrieval. System B: Iterative query refinement."
    )
    
    st.divider()
    st.markdown("**Performance Metrics**")
    st.metric(label="Inference Latency", value=f"{st.session_state.last_latency:.2f} s")
    st.metric(label="Reasoning Steps", value=st.session_state.last_iterations)
    st.caption(f"Backend: Llama-3-8B (Q4_K_M)")

# --- 4. CHAT INTERFACE ---
st.header("Local Legal RAG Benchmark")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "evidence" in msg and msg["evidence"]:
            with st.expander("🔍 View Retrieved Evidence"):
                for chunk in msg["evidence"]:
                    st.info(chunk)

# User Input Logic
if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.time()
        
        if "Traditional" in workflow:
            with st.spinner("System A: Single-pass retrieval..."):
                result = system_a.run(prompt)
                answer = result.answer
                used_chunks = result.used
                st.session_state.last_iterations = 1
        else:
            with st.spinner("System B: Iterative refinement loop..."):
                result = system_b.run(prompt)
                answer = result.answer
                used_chunks = result.used
                st.session_state.last_iterations = result.iterations
                if result.refined_queries:
                    st.caption(f"Refined Search Queries: {', '.join(result.refined_queries)}")

        # Telemetry calculations
        st.session_state.last_latency = time.time() - start_time
        
        # Display Answer
        st.write(answer)
        
        # Display Evidence
        evidence_list = []
        with st.expander("🔍 View Retrieved Evidence"):
            for chunk in used_chunks:
                # Handle cases where score might be missing or different naming
                score = getattr(chunk, 'score', 0.0)
                path = getattr(chunk, 'doc_path', 'Unknown Source')
                
                evidence_text = f"**Source:** {path} | **Relevance:** {score:.4f}\n\n> {chunk.text.strip()}"
                evidence_list.append(evidence_text)
                st.info(evidence_text)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "evidence": evidence_list
        })
        
        st.rerun()