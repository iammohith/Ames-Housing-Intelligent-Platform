"""
Knowledge Builder — Generates plain-English documents from pipeline artifacts,
chunks and embeds them, and loads into ChromaDB.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List

from core.metrics import knowledge_base_chunks_total


class KnowledgeBuilder:
    """Builds RAG knowledge base from pipeline results."""

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    async def build(self, pipeline_results: Dict[str, Any], run_id: str) -> int:
        documents = self._generate_documents(pipeline_results)
        chunks = self._chunk_documents(documents)

        try:
            chunk_count = self._index_chunks(chunks, run_id)
            knowledge_base_chunks_total.set(chunk_count)
            return chunk_count
        except Exception:
            # ChromaDB may not be available in pipeline container
            # Dashboard container handles this
            self._save_documents_to_disk(documents, run_id)
            chunk_count = len(chunks)
            knowledge_base_chunks_total.set(chunk_count)
            return chunk_count

    def _generate_documents(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        docs = []

        # 1. Neighborhood stats
        docs.append(
            {
                "title": "neighborhood_stats",
                "content": self._gen_neighborhood_stats(results),
            }
        )
        # 2. Feature importance
        docs.append(
            {
                "title": "feature_importance_report",
                "content": self._gen_feature_importance(results),
            }
        )
        # 3. Model evaluation
        docs.append(
            {
                "title": "model_evaluation_report",
                "content": self._gen_model_eval(results),
            }
        )
        # 4. Cleaning report
        docs.append(
            {"title": "cleaning_report", "content": self._gen_cleaning(results)}
        )
        # 5. Anomaly report
        docs.append({"title": "anomaly_report", "content": self._gen_anomaly(results)})
        # 6. Price trends
        docs.append(
            {"title": "price_trends_report", "content": self._gen_price_trends(results)}
        )
        # 7. Pipeline summary
        docs.append(
            {
                "title": "pipeline_summary",
                "content": self._gen_pipeline_summary(results),
            }
        )
        # 8. Feature manifest
        docs.append(
            {
                "title": "feature_manifest",
                "content": self._gen_feature_manifest(results),
            }
        )
        # 9. Market segments
        docs.append(
            {"title": "market_segments", "content": self._gen_market_segments(results)}
        )
        # 10. Data dictionary
        docs.append(
            {"title": "data_dictionary", "content": self._gen_data_dictionary()}
        )

        return docs

    def _gen_neighborhood_stats(self, results) -> str:
        text = "Neighborhood Statistics for Ames, Iowa Housing Market\n\n"
        text += "The Ames Housing dataset contains 28 neighborhoods. "
        text += "Neighborhoods with highest median prices include Northridge Heights (NridgHt), "
        text += "Stone Brook (StoneBr), and Northridge (NoRidge). "
        text += "The most affordable neighborhoods include Meadow Village (MeadowV), "
        text += "Briardale (BrDale), and Iowa DOT/Rail Road (IDOTRR). "
        text += "The price range spans from approximately $12,789 to $755,000. "
        ml_result = results.get("ml_agent")
        if ml_result:
            text += f"The best model achieved R²={getattr(ml_result, 'best_test_r2', 'N/A')} on test data."
        return text

    def _gen_feature_importance(self, results) -> str:
        text = "Feature Importance Report\n\n"
        text += "The top features influencing house prices in Ames, Iowa:\n"
        text += "1. Overall Quality (OverallQual) - Most important predictor\n"
        text += "2. Total Square Footage (TotalSF) - Combined living area\n"
        text += "3. Above Ground Living Area (GrLivArea)\n"
        text += "4. Garage Area - Size of garage\n"
        text += "5. Total Basement SF - Basement square footage\n"
        text += "6. Year Built - Age of the house\n"
        text += "7. Full Bathrooms - Number of full baths\n"
        text += "8. TotalBathrooms - Combined bath count\n"
        text += "9. Fireplaces - Number of fireplaces\n"
        text += "10. Kitchen Quality - Kitchen condition rating\n"
        return text

    def _gen_model_eval(self, results) -> str:
        text = "Model Evaluation Report\n\n"
        text += "Three models were trained and evaluated:\n\n"
        ml_result = results.get("ml_agent")
        if ml_result and hasattr(ml_result, "model_results"):
            for mr in ml_result.model_results:
                text += f"- {mr.model_name}: Test RMSE=${mr.test_metrics.rmse_dollars:,.0f}, "
                text += f"R²={mr.test_metrics.r2:.3f}, MAE=${mr.test_metrics.mae_dollars:,.0f}\n"
            text += f"\nBest model: {getattr(ml_result, 'best_model_name', 'unknown')}"
        return text

    def _gen_cleaning(self, results) -> str:
        text = "Data Cleaning Report\n\n"
        cleaning = results.get("cleaning_agent")
        if cleaning and hasattr(cleaning, "cleaning_report"):
            r = cleaning.cleaning_report
            text += f"Rows in: {r.rows_in}, Rows out: {r.rows_out}\n"
            text += f"Rows dropped: {r.rows_dropped}\n"
            text += f"Structural NA fills across {len(r.structural_na_fills)} columns\n"
            text += f"Post-clean null rate: {r.post_clean_null_rate}\n"
        else:
            text += "14 columns had structural NAs filled with 'None'. "
            text += "LotFrontage imputed with neighborhood median. "
            text += "1 row dropped due to null Electrical value."
        return text

    def _gen_anomaly(self, results) -> str:
        text = "Anomaly Detection Report\n\n"
        anomaly = results.get("anomaly_agent")
        if anomaly and hasattr(anomaly, "anomaly_report"):
            r = anomaly.anomaly_report
            text += f"Total flagged: {r.total_flagged} ({r.pct_of_dataset:.1f}% of dataset)\n"
            text += f"Isolation Forest flags: {r.isolation_forest_flags}\n"
            text += f"Z-score flags: {r.zscore_flags}\n"
            text += f"Both methods: {r.both_methods_flags}\n"
        else:
            text += "Anomaly detection used Isolation Forest and Z-score analysis."
        return text

    def _gen_price_trends(self, results) -> str:
        return (
            "Price Trends Report\n\nThe dataset covers sales from 2006 to 2010. "
            "Average prices peaked in 2007 before the financial crisis impact. "
            "Seasonal patterns show higher sales in spring/summer months (May-July). "
            "2010 data served as the holdout test set for model evaluation."
        )

    def _gen_pipeline_summary(self, results) -> str:
        text = "Pipeline Execution Summary\n\n"
        text += "The Ames Housing Intelligence Platform processes 2,930 properties through 8 agents:\n"
        text += "1. Ingestion - CSV validation and hash computation\n"
        text += "2. Schema Validation - Column type classification\n"
        text += "3. Data Cleaning - Null handling and standardization\n"
        text += "4. Feature Engineering - 12 domain features created\n"
        text += "5. Encoding & Scaling - Ordinal, target, and one-hot encoding\n"
        text += "6. Anomaly Detection - Isolation Forest + Z-score\n"
        text += "7. ML Training - Ridge, XGBoost, LightGBM comparison\n"
        text += "8. Orchestration - Knowledge base and audit trail\n"
        return text

    def _gen_feature_manifest(self, results) -> str:
        text = "Feature Manifest\n\n12 engineered features:\n"
        text += "- TotalSF: Total square footage (basement + 1st + 2nd floor)\n"
        text += "- PorchSF: Combined porch/deck area\n"
        text += "- HouseAge: Years since built (relative to sale year)\n"
        text += "- RemodAge: Years since last remodel\n"
        text += "- GarageAge: Years since garage built\n"
        text += "- TotalBathrooms: Full + half baths combined\n"
        text += "- HasPool, HasGarage, HasBasement, HasFireplace, IsNew: Binary flags\n"
        text += "- OverallScore: Quality × Condition interaction\n"
        return text

    def _gen_market_segments(self, results) -> str:
        return (
            "Market Segments Report\n\nAmes housing market segments:\n"
            "- Luxury: NridgHt, StoneBr, NoRidge (median >$250k)\n"
            "- Upper-mid: Timber, Veenker, Somerst (median $180-250k)\n"
            "- Mid-market: CollgCr, Crawfor, Gilbert (median $150-180k)\n"
            "- Affordable: Edwards, OldTown, BrDale (median <$130k)\n"
            "- Budget: MeadowV, IDOTRR, BrkSide (median <$110k)"
        )

    def _gen_data_dictionary(self) -> str:
        return (
            "Ames Housing Data Dictionary\n\n"
            "The dataset contains 82 columns describing 2,930 residential properties.\n"
            "Key columns: SalePrice (target), Overall Qual (1-10 quality rating), "
            "Gr Liv Area (above ground living area), Neighborhood (28 areas), "
            "Year Built, Total Bsmt SF, Garage Area, Full Bath, Kitchen Qual.\n"
            "The data spans sales from 2006-2010 in Ames, Iowa."
        )

    def _chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        chunks = []
        for doc in documents:
            content = doc["content"]
            words = content.split()
            for i in range(0, len(words), self.CHUNK_SIZE - self.CHUNK_OVERLAP):
                chunk_words = words[i : i + self.CHUNK_SIZE]
                if chunk_words:
                    chunks.append(
                        {
                            "title": doc["title"],
                            "content": " ".join(chunk_words),
                            "chunk_index": len(chunks),
                        }
                    )
        return chunks

    def _index_chunks(self, chunks, run_id: str) -> int:
        import chromadb
        from chromadb.utils.embedding_functions import \
            SentenceTransformerEmbeddingFunction

        emb_fn = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path="/app/chroma")
        collection = client.get_or_create_collection(
            "ames_knowledge", embedding_function=emb_fn
        )
        # Reset collection
        try:
            client.delete_collection("ames_knowledge")
        except Exception:
            pass
        collection = client.create_collection(
            "ames_knowledge", embedding_function=emb_fn
        )

        ids = [f"{run_id}_{c['chunk_index']}" for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [{"title": c["title"], "run_id": run_id} for c in chunks]

        if documents:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)

    def _save_documents_to_disk(self, documents, run_id: str):
        doc_dir = f"/app/artifacts/knowledge/{run_id}"
        os.makedirs(doc_dir, exist_ok=True)
        for doc in documents:
            with open(os.path.join(doc_dir, f"{doc['title']}.txt"), "w") as f:
                f.write(doc["content"])

    def rebuild_from_artifacts(
        self, artifacts_dir: str = "/app/artifacts/knowledge"
    ) -> int:
        if not os.path.exists(artifacts_dir):
            return 0
        documents = []
        runs = os.listdir(artifacts_dir)
        if not runs:
            return 0

        runs_with_time = [
            (r, os.path.getmtime(os.path.join(artifacts_dir, r))) for r in runs
        ]
        runs_with_time.sort(key=lambda x: x[1], reverse=True)
        run_dir = os.path.join(artifacts_dir, runs_with_time[0][0])

        if not os.path.isdir(run_dir):
            return 0

        for f in os.listdir(run_dir):
            if f.endswith(".txt"):
                with open(os.path.join(run_dir, f)) as fh:
                    documents.append(
                        {"title": f.replace(".txt", ""), "content": fh.read()}
                    )

        if not documents:
            return 0

        chunks = self._chunk_documents(documents)
        return self._index_chunks(chunks, "manual-rebuild")
