"""
CreditLens-AI
Database Connection Test
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from src.database.connection import AsyncSessionLocal


async def main() -> None:

    print("=" * 70)
    print("CreditLens-AI - Database Connection Test")
    print("=" * 70)

    async with AsyncSessionLocal() as session:

        result = await session.execute(
            text("SELECT current_database();")
        )

        database_name = result.scalar_one()

        print()
        print(f"Connected database : {database_name}")

        if database_name != "creditlens_ai":
            raise RuntimeError(
                "Connected to the wrong database."
            )

        print("✓ PostgreSQL connection successful")
        print("✓ Correct database confirmed")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())