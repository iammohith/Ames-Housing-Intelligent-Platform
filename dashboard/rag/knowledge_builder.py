"""
Dashboard Knowledge Builder — mirror of pipeline's builder for re-indexing.
"""

from __future__ import annotations

import os


class KnowledgeBuilder:
    """Re-indexes knowledge base from saved artifacts."""

    def rebuild_from_artifacts(
        self, artifacts_dir: str = "/app/artifacts/knowledge"
    ) -> int:
        if not os.path.exists(artifacts_dir):
            return 0

        documents = []
        for run_dir in sorted(os.listdir(artifacts_dir), reverse=True)[:1]:
            doc_dir = os.path.join(artifacts_dir, run_dir)
            if not os.path.isdir(doc_dir):
                continue
            for f in os.listdir(doc_dir):
                if f.endswith(".txt"):
                    with open(os.path.join(doc_dir, f)) as fh:
                        documents.append(
                            {"title": f.replace(".txt", ""), "content": fh.read()}
                        )

        if not documents:
            return 0

        # Chunk documents
        chunks = []
        for doc in documents:
            words = doc["content"].split()
            for i in range(0, len(words), 462):  # 512 - 50 overlap
                chunk_words = words[i : i + 512]
                if chunk_words:
                    chunks.append(
                        {"title": doc["title"], "content": " ".join(chunk_words)}
                    )

        # Index into ChromaDB
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            # Use same model name as pipeline's knowledge builder for consistency
            emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            client = chromadb.PersistentClient(path="/app/chroma")
            try:
                client.delete_collection("ames_knowledge")
            except Exception:
                pass
            collection = client.create_collection(
                "ames_knowledge", embedding_function=emb_fn
            )

            ids = [f"rebuild_{i}" for i in range(len(chunks))]
            docs = [c["content"] for c in chunks]
            metadatas = [{"title": c["title"]} for c in chunks]

            if docs:
                collection.add(ids=ids, documents=docs, metadatas=metadatas)
            return len(chunks)
        except Exception:
            return 0
