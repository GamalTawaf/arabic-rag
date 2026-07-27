"""query cache is scoped to the pipeline that produced the answer

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-26

Adds ``pipeline_key``. Before it, the cache key was (model_key, embedding,
guard) — the retrieval config, the depths and the generating model were absent,
so a question asked under one config returned the answer another config had
produced.

Existing rows cannot be attributed to a pipeline after the fact, and guessing
would re-introduce exactly the wrong-answer bug this migration exists to close.
They are deleted: the cache is a derived, rebuildable artifact with no TTL, and
losing it costs one regeneration per question.
"""

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM query_cache")
    op.add_column("query_cache", sa.Column("pipeline_key", sa.String(), nullable=False))
    op.drop_index("ix_query_cache_model_key", table_name="query_cache")
    # Both columns, in the order lookup() filters them.
    op.create_index(
        "ix_query_cache_scope", "query_cache", ["model_key", "pipeline_key"]
    )


def downgrade() -> None:
    op.drop_index("ix_query_cache_scope", table_name="query_cache")
    op.create_index("ix_query_cache_model_key", "query_cache", ["model_key"])
    op.drop_column("query_cache", "pipeline_key")
