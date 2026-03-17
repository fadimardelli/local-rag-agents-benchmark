import streamlit as st
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="LexLocal | Thesis UI",
    page_icon="⚖️",
    layout="wide"
)

# --- Initialize Chat History in Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Controls & Telemetry ---
with st.sidebar:
    st.title("LexLocal")
    st.subheader("Resource-Aware Legal Assistant")
    st.divider()
    
    # 1. Document Upload
    st.markdown("**1. Ingestion**")
    uploaded_file = st.file_uploader("Upload Contract (PDF/TXT)", type=["pdf", "txt"])
    if uploaded_file:
        st.success(f"Indexed: {uploaded_file.name}")
    
    st.divider()
    
    # 2. Workflow Selection
    st.markdown("**2. Workflow Strategy**")
    workflow = st.radio(
        "Select Pipeline:",
        ["⚡ Traditional RAG (System A)", "🧠 Agentic Loop (System B)"],
        help="System A is a linear single-shot search. System B uses an iterative reflection loop."
    )
    
    st.divider()
    
    # 3. Hardware Telemetry (Metrics)
    st.markdown("**3. Hardware Telemetry**")
    st.caption("Live Local Metrics (Llama-3-8B Q4_K_M)")
    
    # In a real app, these values would update dynamically from your backend
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="VRAM Usage", value="5.8 GB", delta="-0.2 GB (stable)", delta_color="inverse")
    with col2:
        st.metric(label="RAM Usage", value="11.2 GB")
        
    st.metric(label="Decode Speed", value="12.4 tok/s")
    st.metric(label="Time to First Token", value="3.2 s" if "Traditional" in workflow else "Pending...")

# --- Main Chat Interface ---
st.header("Legal Document Q&A")

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question about the liability clause..."):
    # 1. Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Response Logic
    with st.chat_message("assistant"):
        
        # --- IF TRADITIONAL RAG (SYSTEM A) ---
        if "Traditional" in workflow:
            with st.spinner("Retrieving and generating (Linear)..."):
                time.sleep(1.5) # Simulate fast TTFT
                response = "Based on the retrieved context, the liability clause limits damages to the total amount paid under the contract. *(Note: This is a single-shot retrieval.)*"
                st.write(response)
                
        # --- IF AGENTIC RAG (SYSTEM B) ---
        else:
            # The "Glass Box" Thought Process
            with st.status("Agentic Loop: Reasoning...", expanded=True) as status:
                st.write("🔍 **Iteration 1:** Retrieving chunks for 'liability clause'...")
                time.sleep(1)
                st.write("🤔 **Reflection:** Found general liability limit. *Missing exceptions for gross negligence.*")
                time.sleep(1)
                st.write("✍️ **Query Reformulation:** Searching for 'gross negligence exceptions to liability'...")
                time.sleep(1)
                st.write("🔍 **Iteration 2:** Retrieved Section 4.2 (Exceptions).")
                time.sleep(1)
                st.write("✅ **Reflection:** Sufficient context gathered. Generating final answer.")
                status.update(label="Reasoning Complete (2 Iterations)", state="complete", expanded=False)
            
            # Final Generated Answer
            response = "Based on Section 4.1, the liability is generally limited to the contract value. **However**, according to the exception found in Section 4.2, this limitation does *not* apply in cases of gross negligence or willful misconduct."
            st.write(response)

        # 3. Save assistant message to state
        st.session_state.messages.append({"role": "assistant", "content": response})