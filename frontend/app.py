"""
DocMind Frontend
Streamlit UI for the Agentic RAG system.
"""

import streamlit as st
import requests
import json
import time

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="DocMind — Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "agent_thinking" not in st.session_state:
    st.session_state.agent_thinking = False

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def check_api_health():
    """Check if the backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def upload_document(file):
    """Upload a document to the backend."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(
            f"{API_URL}/upload",
            files=files,
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}

def ask_question(question: str):
    """Send a question to the agent."""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}

def get_documents():
    """Get list of uploaded documents."""
    try:
        response = requests.get(f"{API_URL}/documents", timeout=10)
        return response.json()
    except Exception:
        return {"documents": [], "total_chunks": 0}

def clear_all_documents():
    """Clear all documents from the system."""
    try:
        response = requests.delete(f"{API_URL}/documents", timeout=30)
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}

def clear_memory():
    """Clear agent conversation memory."""
    try:
        response = requests.post(f"{API_URL}/clear-memory", timeout=10)
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}

def get_confidence_color(score):
    """Return color based on confidence score."""
    if score >= 0.7:
        return "🟢"
    elif score >= 0.4:
        return "🟡"
    else:
        return "🔴"

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.title("🧠 DocMind")
    st.caption("Agentic RAG Research Assistant")
    st.divider()

    # API health check
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Offline")
        st.info("Run: uvicorn backend.main:app --reload --port 8000")
        st.stop()

    st.divider()

    # document upload section
    st.subheader("📄 Upload Documents")
    st.caption("Supported: PDF, TXT, DOCX (max 50MB)")

    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # check if already uploaded
            already_uploaded = any(
                d["name"] == uploaded_file.name
                for d in st.session_state.uploaded_docs
            )

            if not already_uploaded:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    result = upload_document(uploaded_file)

                if result.get("success"):
                    st.success(
                        f"✅ {uploaded_file.name} — "
                        f"{result.get('chunks_created', 0)} chunks"
                    )
                    st.session_state.uploaded_docs.append({
                        "name": uploaded_file.name,
                        "chunks": result.get("chunks_created", 0),
                        "file_id": result.get("file_id", "")
                    })
                else:
                    st.error(
                        f"❌ {uploaded_file.name}: "
                        f"{result.get('message', 'Upload failed')}"
                    )

    st.divider()

    # document library
    st.subheader("📚 Document Library")
    docs_data = get_documents()
    total_chunks = docs_data.get("total_chunks", 0)

    if st.session_state.uploaded_docs:
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"📄 **{doc['name']}**")
            st.caption(f"{doc['chunks']} chunks indexed")
    else:
        st.caption("No documents uploaded yet")

    if total_chunks > 0:
        st.info(f"📊 Total chunks in store: {total_chunks}")

    st.divider()

    # controls
    st.subheader("⚙️ Controls")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Docs", use_container_width=True):
            with st.spinner("Clearing..."):
                result = clear_all_documents()
            if result.get("success"):
                st.session_state.uploaded_docs = []
                st.success("Cleared")
                st.rerun()

    with col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            clear_memory()
            st.success("Cleared")
            st.rerun()

    st.divider()
    st.caption("Built with Haystack + Groq + Qdrant")


# ─────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────

st.title("🧠 DocMind Research Assistant")
st.caption(
    "Upload documents and ask questions. "
    "The agent searches your documents, browses the web, "
    "and shows its full reasoning."
)

# show info if no documents uploaded
if not st.session_state.uploaded_docs:
    st.info(
        "👈 Upload documents from the sidebar to get started. "
        "Supported formats: PDF, TXT, DOCX"
    )

st.divider()

# ─────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # show reasoning trace for assistant messages
        if message["role"] == "assistant" and "trace" in message:
            with st.expander(
                f"🔍 Agent Reasoning ({message.get('iterations', 0)} steps)",
                expanded=False
            ):
                for step in message["trace"]:
                    step_type = step.get("type", "")

                    if step_type == "thought":
                        st.markdown(f"💭 **Thought:** {step['content']}")
                    elif step_type == "action":
                        tool = step.get("tool", "unknown")
                        st.markdown(f"🔧 **Tool Used:** `{tool}`")
                        st.caption(f"Input: {step.get('input', '')[:100]}")
                    elif step_type == "observation":
                        tool = step.get("tool", "unknown")
                        st.markdown(f"👁️ **Observation from** `{tool}`:")
                        st.code(
                            step.get("content", "")[:300],
                            language="text"
                        )

        # show sources
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("📎 Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")

        # show timing
        if message["role"] == "assistant" and "time_taken" in message:
            st.caption(
                f"⏱️ {message['time_taken']}s · "
                f"{message.get('iterations', 0)} agent iterations"
            )

# ─────────────────────────────────────────
# QUESTION INPUT
# ─────────────────────────────────────────

question = st.chat_input(
    "Ask a question about your documents or anything else..."
)

if question:
    # add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # display user message
    with st.chat_message("user"):
        st.markdown(question)

    # get agent response
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            start = time.time()
            result = ask_question(question)
            elapsed = time.time() - start

        if result.get("success"):
            answer = result.get("answer", "No answer received")
            trace = result.get("reasoning_trace", [])
            sources = result.get("sources", [])
            iterations = result.get("iterations", 0)
            time_taken = result.get("time_taken", elapsed)

            # display answer
            st.markdown(answer)

            # show reasoning trace
            if trace:
                with st.expander(
                    f"🔍 Agent Reasoning ({iterations} steps)",
                    expanded=False
                ):
                    for step in trace:
                        step_type = step.get("type", "")

                        if step_type == "thought":
                            st.markdown(
                                f"💭 **Thought:** {step['content']}"
                            )
                        elif step_type == "action":
                            tool = step.get("tool", "unknown")
                            st.markdown(
                                f"🔧 **Tool Used:** `{tool}`"
                            )
                            st.caption(
                                f"Input: {step.get('input', '')[:100]}"
                            )
                        elif step_type == "observation":
                            tool = step.get("tool", "unknown")
                            st.markdown(
                                f"👁️ **Observation from** `{tool}`:"
                            )
                            st.code(
                                step.get("content", "")[:300],
                                language="text"
                            )

            # show sources
            if sources:
                with st.expander("📎 Sources", expanded=False):
                    for source in sources:
                        st.markdown(f"- {source}")

            # show timing
            st.caption(
                f"⏱️ {time_taken}s · {iterations} agent iterations"
            )

            # save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "trace": trace,
                "sources": sources,
                "iterations": iterations,
                "time_taken": time_taken
            })

        else:
            error_msg = result.get(
                "message",
                "Something went wrong. Please try again."
            )
            st.error(f"❌ {error_msg}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {error_msg}"
            })