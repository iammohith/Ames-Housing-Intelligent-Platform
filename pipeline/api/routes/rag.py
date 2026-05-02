"""
RAG API Routes — knowledge base management.
"""
from __future__ import annotations
from datetime import datetime
from fastapi import APIRouter, Depends
from core.schemas import KnowledgeBaseStatus
from api.middleware import verify_api_key
router = APIRouter()


@router.post("/rebuild-knowledge-base")
async def rebuild_knowledge_base(api_key: str = Depends(verify_api_key)):
    """Re-chunk and re-index all artifacts into ChromaDB."""
    try:
        from core.knowledge_builder import KnowledgeBuilder
        kb = KnowledgeBuilder()
        chunks = await kb.build({}, "manual-rebuild")
        return {"message": f"Knowledge base rebuilt: {chunks} chunks indexed"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/knowledge-base/status", response_model=KnowledgeBaseStatus)
async def get_kb_status():
    """Get knowledge base chunk count and metadata."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="/app/chroma")
        collection = client.get_collection("ames_knowledge")
        count = collection.count()
        return KnowledgeBaseStatus(
            chunk_count=count,
            document_count=10,
            last_updated=datetime.utcnow(),
            documents=["neighborhood_stats", "feature_importance_report",
                       "model_evaluation_report", "cleaning_report",
                       "anomaly_report", "price_trends_report",
                       "pipeline_summary", "feature_manifest",
                       "market_segments", "data_dictionary"],
        )
    except Exception:
        return KnowledgeBaseStatus()
