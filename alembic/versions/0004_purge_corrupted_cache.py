"""Purge the answer cache once, to evict a corrupted generation.

Qwen2.5-72B emitted one answer containing hiragana and a stray ampersand
("...أماكن العمل،&oその [&المادة 103]"). `cache.store` accepted it — it only
rejected empty answers at the time — and because the cache is append-only with
no TTL, every semantically-equivalent question served those exact bytes back
forever. `cache.foreign_scripts` now refuses to store such an answer; this
migration removes the one already stored.

Deletes the whole table rather than matching the corrupted rows. This is a
cache: every row is regenerable by asking the question again, so a full purge
costs one generation per distinct question and cannot delete anything that
matters, while a regex over Unicode ranges could miss a row or take a good one.

Irreversible by design: `downgrade` is a no-op because restoring a corrupted
cache entry is not a thing anyone wants.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM query_cache")


def downgrade() -> None:
    pass
