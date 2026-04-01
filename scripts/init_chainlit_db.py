"""Create Chainlit data-layer tables in PostgreSQL (Thread, Step, Feedback, …).

Chainlit reads DATABASE_URL and uses asyncpg, but it does not apply migrations.
Run once after enabling persistence:

    python -m scripts.init_chainlit_db

Schema matches https://github.com/Chainlit/chainlit-datalayer (prisma migrations).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import psycopg

from config import settings

_SCHEMA_SQL = Path(__file__).resolve().parent / "chainlit_postgres_schema.sql"


def _iter_statements(sql: str) -> Iterator[str]:
    buf: list[str] = []
    for line in sql.splitlines():
        if line.strip().startswith("--"):
            continue
        if not line.strip() and not buf:
            continue
        buf.append(line)
        if line.strip().endswith(";"):
            stmt = "\n".join(buf).strip()
            buf.clear()
            if stmt:
                yield stmt


def _ensure_metadata_column_defaults(conn: psycopg.Connection) -> None:
    """Chainlit's update_thread drops null keys; omitted metadata must not violate NOT NULL."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = current_schema()
              AND c.relkind = 'r'
              AND c.relname = 'Thread'
            LIMIT 1
            """
        )
        if cur.fetchone() is None:
            return

    stmts = [
        'ALTER TABLE "Thread" ALTER COLUMN "metadata" SET DEFAULT \'{}\'::jsonb',
        'UPDATE "Thread" SET "metadata" = \'{}\'::jsonb WHERE "metadata" IS NULL',
        'ALTER TABLE "User" ALTER COLUMN "metadata" SET DEFAULT \'{}\'::jsonb',
        'UPDATE "User" SET "metadata" = \'{}\'::jsonb WHERE "metadata" IS NULL',
        'ALTER TABLE "Step" ALTER COLUMN "metadata" SET DEFAULT \'{}\'::jsonb',
        'UPDATE "Step" SET "metadata" = \'{}\'::jsonb WHERE "metadata" IS NULL',
        'ALTER TABLE "Element" ALTER COLUMN "metadata" SET DEFAULT \'{}\'::jsonb',
        'UPDATE "Element" SET "metadata" = \'{}\'::jsonb WHERE "metadata" IS NULL',
    ]
    with conn.cursor() as cur:
        for sql in stmts:
            cur.execute(sql)


def _chainlit_schema_exists(conn: psycopg.Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = current_schema()
              AND c.relkind = 'r'
              AND c.relname = 'Thread'
            LIMIT 1
            """
        )
        return cur.fetchone() is not None


def main() -> int:
    if not _SCHEMA_SQL.is_file():
        print(f"Missing schema file: {_SCHEMA_SQL}", file=sys.stderr)
        return 1

    sql_text = _SCHEMA_SQL.read_text(encoding="utf-8")

    with psycopg.connect(settings.postgres_uri, autocommit=True) as conn:
        if not _chainlit_schema_exists(conn):
            with conn.cursor() as cur:
                for stmt in _iter_statements(sql_text):
                    cur.execute(stmt)
            print("Applied Chainlit PostgreSQL schema successfully.")
        _ensure_metadata_column_defaults(conn)

    print("Chainlit DB ready (metadata column defaults applied).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
