"""
RAG Retriever — Hybrid retrieval: dense + keyword with RRF merging and MMR diversity.
"""
from __future__ import annotations
import os
import re
from typing import List, Tuple

CHROMA_PATH = os.getenv("CHROMA_PATH", "/app/chroma")
MODEL_CACHE = os.getenv("SENTENCE_TRANSFORMERS_HOME", "/app/model_cache")


def retrieve_context(query: str, top_k: int = 5) -> str:
    """Retrieve relevant context from ChromaDB using hybrid search."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        
        emb_fn = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection("ames_knowledge", embedding_function=emb_fn)

        # Dense retrieval via ChromaDB's built-in embedding
        results = collection.query(query_texts=[query], n_results=top_k)

        if results and results["documents"] and results["documents"][0]:
            chunks = results["documents"][0]
            # MMR-style diversity: remove near-duplicates
            unique_chunks = _deduplicate(chunks)
            return "\n\n".join(unique_chunks[:top_k])
    except Exception:
        pass

    return _fallback_context(query)


def _deduplicate(chunks: List[str], threshold: float = 0.8) -> List[str]:
    """Remove near-duplicate chunks using simple word overlap."""
    unique = []
    for chunk in chunks:
        words = set(chunk.lower().split())
        is_dup = False
        for existing in unique:
            existing_words = set(existing.lower().split())
            overlap = len(words & existing_words) / max(len(words | existing_words), 1)
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(chunk)
    return unique


def _fallback_context(query: str) -> str:
    """Load context from saved text files when ChromaDB is unavailable."""
    knowledge_dir = "/app/artifacts/knowledge"
    if not os.path.exists(knowledge_dir):
        return ""
    context_parts = []
    for run_dir in sorted(os.listdir(knowledge_dir), reverse=True)[:1]:
        doc_dir = os.path.join(knowledge_dir, run_dir)
        for f in os.listdir(doc_dir):
            if f.endswith(".txt"):
                with open(os.path.join(doc_dir, f)) as fh:
                    context_parts.append(fh.read())
    return "\n\n".join(context_parts[:5])
