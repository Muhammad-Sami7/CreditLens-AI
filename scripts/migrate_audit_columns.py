"""Add audit-log columns to the existing `predictions` table (safe to re-run).

Run from the project root:  python -m scripts.migrate_audit_columns
Existing rows are kept; the new columns are NULL for old assessments.
"""
import asyncio

from sqlalchemy import text

from src.database.connection import engine

STATEMENTS = [
    "ALTER TABLE public.predictions ADD COLUMN IF NOT EXISTS model_version VARCHAR(100) NULL",
    "ALTER TABLE public.predictions ADD COLUMN IF NOT EXISTS model_weights JSONB NULL",
    "ALTER TABLE public.predictions ADD COLUMN IF NOT EXISTS explanation_text TEXT NULL",
    "ALTER TABLE public.predictions ADD COLUMN IF NOT EXISTS feature_completeness DOUBLE PRECISION NULL",
    "ALTER TABLE public.predictions ADD COLUMN IF NOT EXISTS shap_base_value DOUBLE PRECISION NULL",
]


async def main() -> None:
    async with engine.begin() as conn:
        for stmt in STATEMENTS:
            await conn.execute(text(stmt))
        rows = await conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='predictions' ORDER BY ordinal_position"))
        print("predictions columns:", [r[0] for r in rows])
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
