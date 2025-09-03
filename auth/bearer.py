from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings

security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token."""
    raw_token = credentials.credentials
    if raw_token != settings.ai4s.secret_key:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return raw_token
