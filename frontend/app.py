"""
DocMind Frontend — Polished Version
Clean UI with proper error handling and multi-document support.
"""

import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

if "uploaded_files_seen" not in st.session_state:
    st.session_state.uploaded_files_seen = set()

# ─────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────

def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception:
        return False, {}


def upload_doc(file):
    try:
        r = requests.post(
            f"{API_URL}/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=120
        )
        return r.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def ask(question, session_id):
    try:
        r = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "session_id": session_id},
            timeout=120
        )
        return r.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_docs():
    try:
        r = requests.get(f"{API_URL}/documents", timeout=10)
        return r.json()
    except Exception:
        return {"documents": [], "total_chunks": 0, "total_files": 0}


def clear_docs():
    try:
        r = requests.delete(f"{API_URL}/documents", timeout=30)
        return r.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def clear_mem():
    try:
        requests.post(
            f"{API_URL}/clear-memory",
            params={"session_id": st.session_state.session_id},
            timeout=10
        )
    except Exception:
        pass

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.caption("Agentic RAG Research Assistant")
    st.divider()

    # health check
    healthy, health_data = check_health()
    if healthy:
        st.success("Backend connected")
        chunks = health_data.get("documents_indexed", 0)
        if chunks > 0:
            st.caption(f"{chunks} chunks indexed")
    else:
        st.error("Backend offline")
        st.code("uvicorn backend.main:app --reload --port 8000")
        st.stop()

    st.divider()

    # upload section
    st.markdown("#### Upload Documents")
    st.caption("PDF · TXT · DOCX  (max 50MB each)")

    files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed"
    )

    if files:
        for f in files:
            file_key = f"{f.name}_{f.size}"
            if file_key not in st.session_state.uploaded_files_seen:
                with st.spinner(f"Indexing {f.name}..."):
                    result = upload_doc(f)

                if result.get("success"):
                    chunks = result.get("chunks_created", 0)
                    st.success(f"✓ {f.name}  ({chunks} chunks)")
                    st.session_state.uploaded_files_seen.add(file_key)
                else:
                    msg = result.get("message", "Upload failed")
                    st.error(f"✗ {f.name}: {msg}")

    st.divider()

    # document library
    st.markdown("#### Indexed Documents")
    docs_data = get_docs()
    docs = docs_data.get("documents", [])
    total_chunks = docs_data.get("total_chunks", 0)

    if docs:
        for doc in docs:
            st.markdown(f"📄 **{doc['filename']}**")
            st.caption(f"{doc.get('chunks', 0)} chunks")
        st.caption(f"Total: {total_chunks} chunks across {len(docs)} files")
    else:
        st.caption("No documents uploaded yet")

    st.divider()

    # controls
    st.markdown("#### Controls")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Docs", use_container_width=True):
            with st.spinner("Clearing..."):
                result = clear_docs()
            if result.get("success"):
                st.session_state.uploaded_files_seen = set()
                st.rerun()
            else:
                st.error(result.get("message", "Failed"))

    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = f"session_{int(time.time())}"
            clear_mem()
            st.rerun()

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[-8:]}`")

# ─────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────

st.markdown("## DocMind — Research Assistant")
st.caption(
    "Upload documents and ask questions. "
    "The agent searches your documents, browses the web when needed, "
    "and shows its full reasoning trace."
)

if total_chunks == 0:
    st.info("👈 Upload documents from the sidebar to get started.")

st.divider()

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":

            # confidence
            conf = msg.get("confidence", {})
            if conf:
                pct = conf.get("percentage", 0)
                label = conf.get("label", "")
                color = conf.get("color", "orange")
                emoji = {"green": "🟢", "orange": "🟡", "red": "🔴"}.get(color, "⚪")
                st.caption(f"{emoji} Confidence: {label} ({pct}%)")
                if conf.get("warning"):
                    st.warning(conf["warning"])

            # reasoning trace
            trace = msg.get("trace", [])
            if trace:
                with st.expander(f"Reasoning trace ({msg.get('iterations', 0)} steps)"):
                    for step in trace:
                        t = step.get("type", "")
                        if t == "thought":
                            st.markdown(f"💭 **Thought:** {step['content']}")
                        elif t == "action":
                            st.markdown(f"🔧 **Tool:** `{step.get('tool')}`")
                            st.caption(f"Input: {step.get('input','')[:120]}")
                        elif t == "observation":
                            st.markdown(f"👁️ **From** `{step.get('tool')}`:")
                            st.code(step.get("content","")[:400], language="text")

            # sources
            sources = msg.get("sources", [])
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"- {src}")

            # timing
            if "time_taken" in msg:
                st.caption(
                    f"⏱ {msg['time_taken']}s  ·  "
                    f"{msg.get('iterations', 0)} iterations"
                )

# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────

question = st.chat_input("Ask a question about your documents...")

if question:
    # add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(question, st.session_state.session_id)

        if result.get("success"):
            answer = result.get("answer", "No answer received.")
            trace = result.get("reasoning_trace", [])
            sources = result.get("sources", [])
            conf = result.get("confidence", {})
            iterations = result.get("iterations", 0)
            time_taken = result.get("time_taken", 0)

            st.markdown(answer)

            # confidence
            if conf:
                pct = conf.get("percentage", 0)
                label = conf.get("label", "")
                color = conf.get("color", "orange")
                emoji = {"green": "🟢", "orange": "🟡", "red": "🔴"}.get(color, "⚪")
                st.caption(f"{emoji} Confidence: {label} ({pct}%)")
                if conf.get("warning"):
                    st.warning(conf["warning"])

            # reasoning trace
            if trace:
                with st.expander(f"Reasoning trace ({iterations} steps)"):
                    for step in trace:
                        t = step.get("type", "")
                        if t == "thought":
                            st.markdown(f"💭 **Thought:** {step['content']}")
                        elif t == "action":
                            st.markdown(f"🔧 **Tool:** `{step.get('tool')}`")
                            st.caption(f"Input: {step.get('input','')[:120]}")
                        elif t == "observation":
                            st.markdown(f"👁️ **From** `{step.get('tool')}`:")
                            st.code(step.get("content","")[:400], language="text")

            # sources
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"- {src}")

            # timing
            st.caption(f"⏱ {time_taken}s  ·  {iterations} iterations")

            # save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "trace": trace,
                "sources": sources,
                "confidence": conf,
                "iterations": iterations,
                "time_taken": time_taken
            })

        else:
            err = result.get("message", "Something went wrong. Please try again.")
            st.error(f"Error: {err}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {err}"
            })