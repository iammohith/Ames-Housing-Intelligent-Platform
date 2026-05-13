"""
RAG API Routes — knowledge base management.
"""

from __future__ import annotations

from datetime import datetime

from api.middleware import verify_api_key
from core.schemas import KnowledgeBaseStatus
from fastapi import APIRouter, Depends

router = APIRouter()


@router.post("/rebuild-knowledge-base")
def rebuild_knowledge_base(api_key: str = Depends(verify_api_key)):
    """Re-chunk and re-index all artifacts into ChromaDB."""
    try:
        from core.knowledge_builder import KnowledgeBuilder

        kb = KnowledgeBuilder()
        chunks = kb.rebuild_from_artifacts()
        return {"message": f"Knowledge base rebuilt: {chunks} chunks indexed"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/knowledge-base/status", response_model=KnowledgeBaseStatus)
def get_kb_status():
    """Get knowledge base chunk count and metadata."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path="/app/chroma")
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            emb_fn = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
            collection = client.get_collection("ames_knowledge", embedding_function=emb_fn)
            count = collection.count()
        except Exception:
            # Collection doesn't exist yet
            count = 0
        
        return KnowledgeBaseStatus(
            chunk_count=count,
            document_count=10 if count > 0 else 0,
            last_updated=datetime.utcnow() if count > 0 else None,
            documents=[
                "neighborhood_stats",
                "feature_importance_report",
                "model_evaluation_report",
                "cleaning_report",
                "anomaly_report",
                "price_trends_report",
                "pipeline_summary",
                "feature_manifest",
                "market_segments",
                "data_dictionary",
            ] if count > 0 else [],
        )
    except Exception as e:
        import structlog
        structlog.get_logger().warning(f"Failed to get KB status: {e}")
        return KnowledgeBaseStatus(
            chunk_count=0,
            document_count=0,
            last_updated=None,
            documents=[],
        )
