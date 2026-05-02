"""
API Middleware — Auth and rate limiting.
"""
from __future__ import annotations
import os
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for mutation endpoints."""
    expected = os.getenv("API_KEY", "changeme")
    if not api_key or api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key
