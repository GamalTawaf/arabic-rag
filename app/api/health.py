from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db

# Annotated form (not `= Depends(...)`) so the dependency isn't a mutable default.
DbSession = Annotated[AsyncSession, Depends(get_db)]

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/db")
async def health_db(db: DbSession):
    await db.execute(text("SELECT 1"))
    return {"status": "ok"}
