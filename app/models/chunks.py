"""Chunk store: one row per corpus chunk, one vector column per benchmarked model.

Keeping every embedding on the same row is what makes the benchmark honest — the
only thing that varies between runs is which column the search reads.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

# Benchmarked embedding models -> the column holding their vectors.
EMBEDDING_COLUMNS: dict[str, str] = {
    "e5": "emb_e5",
    "bge": "emb_bge",
    "openai": "emb_openai",
    "cohere": "emb_cohere",
}

EMBEDDING_DIMS: dict[str, int] = {
    "e5": 1024,  # intfloat/multilingual-e5-large
    "bge": 1024,  # BAAI/bge-m3
    "openai": 3072,  # text-embedding-3-large
    "cohere": 1536,  # cohere embed-v4
}

# ponytail: HNSW ceiling in pgvector is 2000 dims, so emb_openai (3072) gets no
# index — exact scan is fine at corpus scale (a few thousand chunks, single-digit
# ms). Upgrade path when the corpus grows: store it as halfvec(3072) and index
# with halfvec_cosine_ops, or reduce dimensions via the OpenAI `dimensions` param.
HNSW_INDEXED = ("emb_e5", "emb_bge", "emb_cohere")


def _hnsw(column: str) -> Index:
    return Index(
        f"ix_chunks_{column}_hnsw",
        column,
        postgresql_using="hnsw",
        postgresql_ops={column: "vector_cosine_ops"},
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # "doc:article:seq"
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    article: Mapped[str | None] = mapped_column(String, nullable=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)  # original, diacritics kept
    text_normalized: Mapped[str] = mapped_column(Text, nullable=False)  # index form

    # 'simple' config: Postgres ships no Arabic stemmer, and text_normalized has
    # already been through our own normalizer (tatweel/diacritics stripped,
    # alef/ya/ta-marbuta folded), so any further stemming would only add noise.
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', text_normalized)", persisted=True),
        nullable=True,
    )

    emb_e5: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    emb_bge: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    emb_openai: Mapped[list[float] | None] = mapped_column(Vector(3072), nullable=True)
    emb_cohere: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        *(_hnsw(column) for column in HNSW_INDEXED),
    )
