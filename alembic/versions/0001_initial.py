"""initial chunk store

Revision ID: 0001
Revises:
Create Date: 2026-07-24

"""

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(), nullable=False),  # "doc:article:seq"
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("article", sa.String(), nullable=True),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_normalized", sa.Text(), nullable=False),
        # Generated column instead of a trigger: one less moving part, and it can
        # never drift from text_normalized. 'simple' config because Postgres has
        # no Arabic stemmer and text_normalized is already normalized by us.
        sa.Column(
            "tsv",
            TSVECTOR(),
            sa.Computed("to_tsvector('simple', text_normalized)", persisted=True),
            nullable=True,
        ),
        sa.Column("emb_e5", Vector(1024), nullable=True),
        sa.Column("emb_bge", Vector(1024), nullable=True),
        sa.Column("emb_openai", Vector(3072), nullable=True),
        sa.Column("emb_cohere", Vector(1536), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")

    # ponytail: no HNSW index on emb_openai — pgvector caps HNSW at 2000 dims and
    # text-embedding-3-large is 3072. Exact scan is fine at corpus scale (a few
    # thousand chunks). Upgrade path: halfvec(3072) + halfvec_cosine_ops, or ask
    # OpenAI for reduced `dimensions`.
    for column in ("emb_e5", "emb_bge", "emb_cohere"):
        op.create_index(
            f"ix_chunks_{column}_hnsw",
            "chunks",
            [column],
            postgresql_using="hnsw",
            postgresql_ops={column: "vector_cosine_ops"},
        )


def downgrade() -> None:
    op.drop_table("chunks")
