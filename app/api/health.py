from fastapi import APIRouter

from app.lib.health import database_reachable

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/db")
async def health_db():
    await database_reachable()
    return {"status": "ok"}
