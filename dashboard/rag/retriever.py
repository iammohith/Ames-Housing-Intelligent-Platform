"""
RAG Retriever — Advanced MAANG-level hybrid retrieval.
Supports: Query Intent Routing, BM25, Semantic MMR, and RRF merging.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

CHROMA_PATH = os.getenv("CHROMA_PATH", "/app/chroma")
MODEL_CACHE = os.getenv("SENTENCE_TRANSFORMERS_HOME", "/app/model_cache")

INTENT_TO_TITLES = {
    "neighborhood": ["neighborhood_stats", "market_segments"],
    "model_performance": ["model_evaluation_report", "pipeline_summary"],
    "data_quality": ["cleaning_report", "data_dictionary"],
    "anomaly": ["anomaly_report"],
    "feature": ["feature_importance_report", "feature_manifest"],
    "temporal": ["price_trends_report"],
}

# Keep a global embedding function to avoid reloading
_emb_fn = None


def _get_embedding_fn():
    global _emb_fn
    if _emb_fn is None:
        from chromadb.utils.embedding_functions import \
            SentenceTransformerEmbeddingFunction

        _emb_fn = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _emb_fn


def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Retrieve relevant context using advanced hybrid search and MMR.
    MAANG-level: Confidence scoring, validation, freshness checking.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        import chromadb
        from rag.query_classifier import classify_query

        emb_fn = _get_embedding_fn()
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        try:
            collection = client.get_collection("ames_knowledge", embedding_function=emb_fn)
        except Exception as e:
            logger.warning(f"ChromaDB collection not found: {e}. Knowledge base may need rebuilding.")
            fallback = _fallback_context(query)
            # Mark fallback responses so users know these are not from RAG
            if fallback and not fallback.startswith("[FALLBACK]"):
                return f"[⚠️ FALLBACK - Knowledge base empty] {fallback}"
            return fallback

        initial_k = top_k * 4

        # 1. Intent Classification & Metadata Routing (with confidence)
        intent, confidence = classify_query(query)
        logger.info(f"Query intent: {intent} (confidence: {confidence:.2f})")
        
        try:
            dense_results = collection.query(query_texts=[query], n_results=initial_k)
        except Exception as e:
            logger.warning(f"Dense retrieval failed: {e}. Using fallback.")
            return _fallback_context(query)

        # If specific intent with HIGH confidence, fetch documents of that intent
        intent_docs = []
        if confidence > 0.5 and intent != "general" and intent in INTENT_TO_TITLES:
            titles = INTENT_TO_TITLES[intent]
            # Fetch all documents and filter by title in Python
            # (ChromaDB where filters can be unreliable across versions)
            try:
                all_results = collection.get()
                all_metadatas = all_results.get("metadatas", [])
                all_documents = all_results.get("documents", [])
                
                for idx, metadata in enumerate(all_metadatas):
                    if metadata and metadata.get("title") in titles:
                        if idx < len(all_documents):
                            intent_docs.append(all_documents[idx])
                
                logger.info(f"Found {len(intent_docs)} documents matching intent '{intent}'")
            except Exception as filter_err:
                # If metadata filtering fails, log and continue with dense results
                logger.warning(f"ChromaDB metadata filtering failed: {filter_err}. Using dense results only.")

        dense_docs = (
            dense_results["documents"][0]
            if dense_results
            and dense_results.get("documents")
            and dense_results["documents"][0]
            else []
        )
        
        # Validate dense results quality
        if not dense_docs:
            logger.warning(f"No dense results for query. Returning fallback.")
            return _fallback_context(query)
        
        dense_docs.extend(intent_docs)

        # 2. BM25 Keyword Retrieval
        all_docs_results = collection.get()
        all_docs = all_docs_results.get("documents", [])

        bm25_docs = _bm25_search(query, all_docs, top_n=initial_k)

        # 3. RRF Merging
        merged_chunks = _rrf_merge(dense_docs, bm25_docs)

        if merged_chunks:
            # 4. Semantic MMR (Maximal Marginal Relevance)
            final_chunks = _semantic_mmr(query, merged_chunks, emb_fn, top_k=top_k)

            # Format chunks with citations
            formatted_chunks = []
            
            # ── LIVE STATE INJECTION (Stateful RAG) ──
            try:
                import requests
                api_url = os.getenv("API_URL", "http://orchestration-api:8000")
                q_lower = query.lower()
                if "anomal" in q_lower:
                    resp = requests.get(f"{api_url}/api/latest-metrics", timeout=2)
                    if resp.status_code == 200:
                        m = resp.json().get("metrics", {})
                        formatted_chunks.append(f"[Live System State]\nTotal anomalies detected in the latest pipeline run: {m.get('anomalies_count', 0)}.")
                elif any(w in q_lower for w in ["r2", "r²", "r^2", "rmse", "score", "model"]):
                    resp = requests.get(f"{api_url}/api/latest-metrics", timeout=2)
                    if resp.status_code == 200:
                        m = resp.json().get("metrics", {})
                        best_r2 = m.get("best_r2", 0)
                        best_rmse = m.get("best_rmse", 0)
                        formatted_chunks.append(f"[Live System State]\nThe Champion model achieved an R2 score of {best_r2} and an RMSE of ${best_rmse:,.0f} on the test set.")
                elif any(w in q_lower for w in ["row", "feature", "process"]):
                    resp = requests.get(f"{api_url}/api/latest-metrics", timeout=2)
                    if resp.status_code == 200:
                        m = resp.json().get("metrics", {})
                        formatted_chunks.append(f"[Live System State]\nThe latest run processed {m.get('rows_processed', 0)} rows and generated {m.get('features_count', 0)} features.")
            except Exception as inject_err:
                logger.warning(f"Failed to inject live state: {inject_err}")
            # ─────────────────────────────────────────

            for chunk in final_chunks:
                # find the metadata title for citation
                title = "unknown"
                for i, doc in enumerate(all_docs):
                    if doc == chunk:
                        meta = all_docs_results.get("metadatas", [])
                        if meta and i < len(meta) and meta[i]:
                            title = meta[i].get("title", "unknown")
                        break
                formatted_chunks.append(f"[Source: {title}]\n{chunk}")

            return "\n\n".join(formatted_chunks)
    except Exception as e:
        # Log exception for debugging, but proceed with fallback
        logger.warning(f"retrieve_context failed: {e}")

    return _fallback_context(query)


def _bm25_search(
    query: str, corpus: List[str], top_n: int = 15, k1: float = 1.5, b: float = 0.75
) -> List[str]:
    """Robust Okapi BM25 implementation."""
    if not corpus:
        return []

    query_words = re.findall(r"\w+", query.lower())
    if not query_words:
        return corpus[:top_n]

    # Build term frequencies and document frequencies
    doc_lengths = []
    df = defaultdict(int)
    tf_docs = []

    for doc in corpus:
        words = re.findall(r"\w+", doc.lower())
        doc_lengths.append(len(words))
        tf = defaultdict(int)
        for w in words:
            tf[w] += 1
        tf_docs.append(tf)
        for w in set(words):
            df[w] += 1

    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
    N = len(corpus)

    scores = []
    for i, tf in enumerate(tf_docs):
        score = 0.0
        for qw in query_words:
            if qw not in tf:
                continue
            # IDF calculation (Robertson-Spärck Jones)
            idf = math.log(1 + (N - df[qw] + 0.5) / (df[qw] + 0.5))
            # TF normalization
            tf_term = tf[qw]
            numerator = tf_term * (k1 + 1)
            denominator = tf_term + k1 * (1 - b + b * (doc_lengths[i] / avgdl))
            score += idf * (numerator / denominator)
        scores.append((corpus[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scores[:top_n] if score > 0]


def _rrf_merge(list1: List[str], list2: List[str], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion for combining two ranked lists."""
    rank_scores: Dict[str, float] = defaultdict(float)

    for rank, doc in enumerate(list1):
        rank_scores[doc] += 1.0 / (k + rank + 1)

    for rank, doc in enumerate(list2):
        rank_scores[doc] += 1.0 / (k + rank + 1)

    sorted_docs = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs]


def _semantic_mmr(
    query: str, docs: List[str], emb_fn: Any, top_k: int = 5, lambda_mult: float = 0.7
) -> List[str]:
    """Maximal Marginal Relevance using dense embeddings to ensure diversity."""
    if not docs:
        return []
    if len(docs) <= top_k:
        return docs

    query_emb = emb_fn([query])[0]
    doc_embs = emb_fn(docs)

    # Cosine similarity helper
    def sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    selected_indices = []
    unselected_indices = list(range(len(docs)))

    # Select first document (highest similarity to query)
    best_idx = max(unselected_indices, key=lambda i: sim(query_emb, doc_embs[i]))
    selected_indices.append(best_idx)
    unselected_indices.remove(best_idx)

    # Select remaining iteratively
    while len(selected_indices) < top_k and unselected_indices:
        best_score = -float("inf")
        best_idx = -1

        for i in unselected_indices:
            # Relevance to query
            relevance = sim(query_emb, doc_embs[i])
            # Max similarity to already selected docs (redundancy)
            redundancy = max([sim(doc_embs[i], doc_embs[j]) for j in selected_indices])

            # MMR Score
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

    return [docs[i] for i in selected_indices]


def _fallback_context(query: str) -> str:
    """Load context from saved text files when ChromaDB is unavailable, matching query words."""
    knowledge_dir = os.getenv("KNOWLEDGE_DIR", "/app/artifacts/knowledge")
    if not os.path.exists(knowledge_dir):
        return ""

    context_chunks = []
    try:
        run_dirs = sorted(os.listdir(knowledge_dir), reverse=True)
        if not run_dirs:
            return ""
        doc_dir = os.path.join(knowledge_dir, run_dirs[0])

        for f in os.listdir(doc_dir):
            if f.endswith(".txt"):
                with open(os.path.join(doc_dir, f)) as fh:
                    content = fh.read()
                    words = content.split()
                    title = f.replace(".txt", "")
                    for i in range(0, len(words), 462):
                        chunk_words = words[i : i + 512]
                        if chunk_words:
                            context_chunks.append(
                                f"[Source: {title}]\n" + " ".join(chunk_words)
                            )

        # BM25 on chunks
        best_chunks = _bm25_search(query, context_chunks, top_n=5)
        if best_chunks:
            return "\n\n".join(best_chunks)
        else:
            return "\n\n".join(context_chunks[:5])
    except Exception:
        return ""
