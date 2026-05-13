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

if "conversation" not in st.session_state:
    from rag.conversation import ConversationManager
    st.session_state.conversation = ConversationManager()

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
        if "conversation" in st.session_state:
            st.session_state.conversation.clear()
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

        # Use the ConversationManager to format history
        formatted_history = st.session_state.conversation.format_for_prompt(window_size=5)

        # Retrieve context (returns a dictionary now)
        retrieval_result = retrieve_context(question)
        context = retrieval_result.get("context", "")
        intent = retrieval_result.get("intent", "general")
        confidence = retrieval_result.get("confidence", 0.0)
        fallback = retrieval_result.get("fallback", False)

        # Generate answer
        answer = generate_answer(question, context, chat_history=formatted_history)

        if not answer:
            return "I couldn't generate an answer. Please run the pipeline first to build the knowledge base."

        # Construct provenance metadata badge
        provenance = (
            f"\n\n---\n"
            f"*Source: Local RAG pipeline (flan-t5-base + ChromaDB)*<br>"
            f"*Intent: **{intent}** (Confidence: {confidence:.2f})*<br>"
        )
        if fallback:
            provenance += f"*Fallback used: **Yes***"

        return answer + provenance
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

# ── Check for pending suggested question ──────────────────────────────────────
pending_question = None
if "suggested_question" in st.session_state:
    pending_question = st.session_state.suggested_question
    del st.session_state.suggested_question

# ── Suggested questions (only shown when conversation is empty) ───────────────
suggested_container = st.empty()

if not st.session_state.chat_messages and not pending_question:
    with suggested_container.container():
        st.markdown('<div class="section-title">Suggested Questions</div>', unsafe_allow_html=True)
        q_cols = st.columns(2, gap="medium")
        for i, q in enumerate(SUGGESTED):
            with q_cols[i % 2]:
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state.suggested_question = q
        st.markdown("<br>", unsafe_allow_html=True)
    
    # If a button was clicked in this run, rerun immediately to process it
    if "suggested_question" in st.session_state:
        st.rerun()
else:
    # Explicitly clear the container so stale buttons don't linger during the LLM generation
    suggested_container.empty()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about the Ames housing dataset…")

if pending_question:
    user_input = pending_question

if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    st.session_state.conversation.add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            answer = _get_rag_answer(user_input)
            st.markdown(answer, unsafe_allow_html=True)
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    st.session_state.conversation.add_message("assistant", answer)
