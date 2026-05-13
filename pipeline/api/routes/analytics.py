"""
Analytics API Routes — data exploration and monitoring endpoints.
"""

from __future__ import annotations

import os

import pandas as pd
from fastapi import APIRouter

router = APIRouter()


def _get_knowledge_chunk_count() -> int:
    """Query ChromaDB for actual knowledge base chunk count."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path="/app/chroma")
        collection = client.get_collection("ames_knowledge")
        return collection.count()
    except Exception:
        return 0


@router.get("/anomalies")
async def get_anomalies(page: int = 1, per_page: int = 50, severity: str = None):
    """Paginated anomaly log with optional severity filter."""
    try:
        import psycopg2

        # Validate inputs
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 500:
            per_page = 50
        
        conn = psycopg2.connect(os.getenv("DATABASE_URL_SYNC", ""))
        cur = conn.cursor()
        query = "SELECT * FROM anomaly_log"
        params = []
        if severity and severity.upper() in ["HIGH", "MEDIUM", "LOW"]:
            query += " WHERE severity = %s"
            params.append(severity.upper())
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([per_page, (page - 1) * per_page])
        cur.execute(query, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        total_rows = len(rows)
        conn.close()
        return {
            "anomalies": [dict(zip(cols, r)) for r in rows] if cols else [],
            "page": page,
            "per_page": per_page,
            "total_on_page": total_rows,
        }
    except Exception as e:
        import structlog
        structlog.get_logger().warning(f"Failed to fetch anomalies: {e}")
        return {"anomalies": [], "page": page, "error": str(e)[:100]}


@router.get("/schema-history")
async def get_schema_history():
    """Null rates across runs for drift detection."""
    try:
        import psycopg2

        conn = psycopg2.connect(os.getenv("DATABASE_URL_SYNC", ""))
        cur = conn.cursor()
        cur.execute(
            "SELECT run_id, column_name, null_rate, data_type, is_structural_na FROM schema_history ORDER BY created_at DESC LIMIT 500"
        )
        rows = cur.fetchall()
        conn.close()
        return {
            "history": [
                {
                    "run_id": r[0],
                    "column": r[1],
                    "null_rate": float(r[2]) if r[2] is not None else 0.0,
                    "data_type": r[3],
                    "is_structural_na": r[4],
                }
                for r in rows
            ]
            if rows
            else []
        }
    except Exception as e:
        import structlog
        structlog.get_logger().warning(f"Failed to fetch schema history: {e}")
        return {"history": [], "error": str(e)[:100]}


@router.get("/models")
async def get_models():
    """MLflow registered models and metrics."""
    try:
        import psycopg2

        conn = psycopg2.connect(os.getenv("DATABASE_URL_SYNC", ""))
        cur = conn.cursor()
        cur.execute("SELECT * FROM model_results ORDER BY created_at DESC LIMIT 20")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        conn.close()
        return {"models": [dict(zip(cols, r)) for r in rows]}
    except Exception as e:
        return {"models": [], "error": str(e)}


@router.get("/neighborhood-stats")
async def get_neighborhood_stats():
    """Aggregated stats per neighborhood from raw CSV data."""
    csv_path = "/app/data/AmesHousing.csv"
    if not os.path.exists(csv_path):
        return {"stats": [], "message": "Dataset not found"}
    try:
        df = pd.read_csv(csv_path)
        if "Neighborhood" not in df.columns or "SalePrice" not in df.columns:
            return {"stats": [], "message": "Required columns missing"}
        
        stats = (
            df.groupby("Neighborhood")["SalePrice"]
            .agg(["median", "mean", "std", "count", "min", "max"])
            .reset_index()
            .sort_values("median", ascending=False)
        )
        stats.columns = [
            "neighborhood",
            "median_price",
            "mean_price",
            "std_price",
            "transaction_count",
            "min_price",
            "max_price",
        ]
        # Convert to safe numeric types
        for col in ["median_price", "mean_price", "std_price", "min_price", "max_price"]:
            stats[col] = stats[col].fillna(0).astype(float).round(2)
        stats["transaction_count"] = stats["transaction_count"].astype(int)
        
        return {"stats": stats.to_dict(orient="records")}
    except Exception as e:
        import structlog
        structlog.get_logger().warning(f"Failed to compute neighborhood stats: {e}")
        return {"stats": [], "error": str(e)[:100]}


@router.get("/processed-data")
async def get_processed_data(page: int = 1, per_page: int = 50):
    """
    Paginated processed dataset from the latest successful pipeline run.
    Falls back to raw CSV if no pipeline has run yet.
    """
    try:
        import psycopg2
        
        # Try to get latest successful run
        conn = psycopg2.connect(os.getenv("DATABASE_URL_SYNC", ""))
        cur = conn.cursor()
        cur.execute(
            "SELECT run_id FROM pipeline_runs WHERE status='SUCCESS' ORDER BY completed_at DESC LIMIT 1"
        )
        run_row = cur.fetchone()
        conn.close()
        
        df = None
        if run_row:
            run_id = run_row[0]
            # Try to load processed data from artifacts
            artifact_path = f"/app/artifacts/{run_id}/processed_data.csv"
            if os.path.exists(artifact_path):
                df = pd.read_csv(artifact_path)
        
        # Fallback to raw dataset
        if df is None:
            csv_path = "/app/data/AmesHousing.csv"
            if not os.path.exists(csv_path):
                return {"data": [], "page": page, "total": 0, "message": "No data available"}
            df = pd.read_csv(csv_path)
        
        # Paginate
        total_rows = len(df)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_data = df.iloc[start_idx:end_idx]
        
        return {
            "data": page_data.to_dict(orient="records"),
            "page": page,
            "per_page": per_page,
            "total": total_rows,
            "pages": (total_rows + per_page - 1) // per_page,
        }
    except Exception as e:
        return {"data": [], "page": page, "error": str(e)}


@router.get("/latest-metrics")
async def get_latest_metrics():
    """Get overall metrics from the latest successful run."""
    try:
        import psycopg2

        conn = psycopg2.connect(os.getenv("DATABASE_URL_SYNC", ""))
        cur = conn.cursor()

        cur.execute(
            "SELECT run_id, rows_out FROM pipeline_runs WHERE status='SUCCESS' ORDER BY started_at DESC LIMIT 1"
        )
        run_row = cur.fetchone()
        if not run_row:
            return {"metrics": {}}

        run_id, rows_processed = run_row

        cur.execute("SELECT COUNT(*) FROM anomaly_log")
        anomalies_count = cur.fetchone()[0]

        cur.execute(
            "SELECT test_rmse, test_r2 FROM model_results WHERE run_id=%s AND is_best=TRUE",
            (run_id,),
        )
        model_row = cur.fetchone()
        best_rmse, best_r2 = model_row if model_row else (0, 0)

        conn.close()

        # Dynamic feature count from model artifacts
        features_count = 0
        try:
            import json
            cols_path = f"/app/artifacts/models/{run_id}/feature_cols.json"
            if os.path.exists(cols_path):
                with open(cols_path) as f:
                    features_count = len(json.load(f))
        except Exception:
            features_count = 0

        return {
            "metrics": {
                "rows_processed": rows_processed,
                "features_count": features_count,
                "anomalies_count": anomalies_count,
                "best_rmse": best_rmse,
                "best_r2": best_r2,
                "knowledge_chunks": _get_knowledge_chunk_count(),
            }
        }
    except Exception as e:
        return {"metrics": {}, "error": str(e)}
