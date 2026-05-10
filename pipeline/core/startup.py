"""
Startup Configuration Validation — Fail-fast on missing/invalid config.
Ensures platform is properly configured before any processing.
"""

import os
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger()


def validate_environment():
    """
    Validate all required environment variables and test service connectivity.
    Raises RuntimeError on validation failure (fail-fast).
    """
    errors = []

    # 1. Required environment variables
    required_vars = {
        "DATABASE_URL_SYNC": "PostgreSQL sync connection string (postgresql://...)",
        "REDIS_URL": "Redis connection URL (redis://...)",
        "ARTIFACTS_DIR": "Artifact storage directory path",
        "MLFLOW_TRACKING_URI": "MLflow tracking server URI (http://...)",
    }

    for var, desc in required_vars.items():
        value = os.getenv(var)
        if not value:
            errors.append(f"Missing {var}: {desc}")
        else:
            logger.info(f"✓ {var} configured", value_head=value[:50])

    if errors:
        error_msg = "Startup validation failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 2. Test PostgreSQL connectivity
    db_url = os.getenv("DATABASE_URL_SYNC")
    try:
        import psycopg2

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        conn.close()
        logger.info("✓ PostgreSQL connection successful")
    except Exception as e:
        error_msg = f"Cannot connect to PostgreSQL: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 3. Test Redis connectivity
    redis_url = os.getenv("REDIS_URL")
    try:
        import redis

        r = redis.from_url(redis_url)
        r.ping()
        logger.info("✓ Redis connection successful")
    except Exception as e:
        error_msg = f"Cannot connect to Redis: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 4. Validate and create artifact directories
    artifact_dir = os.getenv("ARTIFACTS_DIR", "/app/artifacts")
    try:
        for subdir in ["models", "encoders", "shap"]:
            path = os.path.join(artifact_dir, subdir)
            os.makedirs(path, exist_ok=True)
        logger.info(f"✓ Artifact directories ready", path=artifact_dir)
    except Exception as e:
        error_msg = f"Cannot create artifact directories: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 5. Validate dataset exists
    csv_path = "/app/data/AmesHousing.csv"
    if not os.path.exists(csv_path):
        error_msg = f"Dataset not found: {csv_path}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    logger.info("✓ Dataset file found", path=csv_path)

    # 6. Validate ChromaDB path
    chroma_path = os.getenv("CHROMA_PATH", "/app/chroma")
    try:
        os.makedirs(chroma_path, exist_ok=True)
        logger.info("✓ ChromaDB path ready", path=chroma_path)
    except Exception as e:
        error_msg = f"Cannot create ChromaDB path: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info("✅ All startup checks passed — system ready")
