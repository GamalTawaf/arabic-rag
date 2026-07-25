from fastapi import FastAPI

from app.api import health
from app.config import settings

app = FastAPI(title=settings.service_name)
app.include_router(health.router)
