"""semantic query cache

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-25

"""

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "query_cache",
        sa.Column("id", sa.String(), nullable=False),  # uuid4().hex
        sa.Column("query_normalized", sa.Text(), nullable=False),
        # 1024 dims: the two local embedders (e5, bge) the service actually runs.
        sa.Column("embedding", Vector(1024), nullable=False),
        sa.Column("model_key", sa.String(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("citations", JSONB(), nullable=False),
        sa.Column("hits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_hit_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Every lookup filters on model_key before it does anything else — a vector
    # from another model is not comparable, it is a wrong answer waiting.
    op.create_index("ix_query_cache_model_key", "query_cache", ["model_key"])
    op.create_index(
        "ix_query_cache_embedding_hnsw",
        "query_cache",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_table("query_cache")
