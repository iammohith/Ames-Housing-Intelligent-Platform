"""
Page 3 — AI Insights Chatbot
Fully offline Q&A using flan-t5-base + ChromaDB RAG pipeline.
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://orchestration-api:8000")

from theme import apply_theme
apply_theme()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("chat_messages", []), ("rag_initialized", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Knowledge Base Status**")
    try:
        r = requests.get(f"{API_URL}/api/knowledge-base/status", timeout=5)
        if r.status_code == 200:
            kb = r.json()
            if isinstance(kb, dict):
                chunks = kb.get("chunk_count", 0)
                docs   = kb.get("document_count", 0)
                upd    = kb.get("last_updated", "")
                colour = "#059669" if chunks > 0 else "#D97706"
                st.markdown(
                    f'<span style="color:{colour};font-weight:600;font-size:0.85rem;">● {chunks} chunks indexed</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<span style="font-size:0.82rem;color:#64748B;">{docs} documents</span>', unsafe_allow_html=True)
                if upd:
                    st.markdown(f'<span style="font-size:0.78rem;color:#94A3B8;">Updated {upd[:19]}</span>', unsafe_allow_html=True)
                st.session_state.rag_initialized = chunks > 0
        else:
            st.markdown('<span style="color:#D97706;font-size:0.82rem;">KB not available</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span style="color:#94A3B8;font-size:0.82rem;">Connecting…</span>', unsafe_allow_html=True)

    st.divider()
    if st.button("↺  Re-index Knowledge Base"):
        try:
            api_key = os.getenv("API_KEY", "changeme")
            r = requests.post(
                f"{API_URL}/api/rebuild-knowledge-base",
                headers={"X-API-Key": api_key},
                timeout=30,
            )
            if r.status_code == 200:
                st.success("Knowledge base rebuilt!")
            else:
                st.error("Rebuild failed.")
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("🗑  Clear Conversation"):
        st.session_state.chat_messages = []
        st.rerun()

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="page-header">
    <div class="page-header-title">🤖 AI Insights Chatbot</div>
    <div class="page-header-sub">
        Ask questions about the Ames Housing dataset in plain English ·
        Powered by <strong>flan-t5-base</strong> RAG · Fully offline
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Helper functions ──────────────────────────────────────────────────────────
def _get_rag_answer(question: str) -> str:
    try:
        from rag.generator import generate_answer
        from rag.retriever import retrieve_context

        history_parts = []
        for msg in st.session_state.chat_messages[-6:-1]:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{prefix}: {msg['content']}")

        context = retrieve_context(question)
        answer  = generate_answer(question, context, chat_history="\n".join(history_parts))

        if not answer:
            return "I couldn't generate an answer. Please run the pipeline first to build the knowledge base."

        return answer + "\n\n---\n*Source: Local RAG pipeline (flan-t5-base + ChromaDB)*"
    except Exception as e:
        return f"I encountered an error while trying to answer your question: {e}"


# ── Suggested questions (only shown when conversation is empty) ───────────────
SUGGESTED = [
    "Which neighborhoods have the highest average sale prices?",
    "What are the top 3 features that influence house prices?",
    "How many anomalies were detected in the last pipeline run?",
    "What was the model's R² score on the 2010 test set?",
    "How does house age affect sale price?",
    "What data quality issues were found in the dataset?",
    "Which year had the most home sales between 2006 and 2010?",
    "What is the Isolation Forest contamination rate used?",
]

if not st.session_state.chat_messages:
    st.markdown('<div class="section-title">Suggested Questions</div>', unsafe_allow_html=True)
    q_cols = st.columns(2, gap="medium")
    for i, q in enumerate(SUGGESTED):
        with q_cols[i % 2]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state.chat_messages.append({"role": "user", "content": q})
                answer = _get_rag_answer(q)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about the Ames housing dataset…"):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            answer = _get_rag_answer(prompt)
            st.markdown(answer)
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
