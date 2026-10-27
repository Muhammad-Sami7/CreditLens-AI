import asyncio

from src.database.connection import engine, Base
from src.database import models  # noqa: F401


async def create_tables() -> None:
    print("=" * 70)
    print("CreditLens-AI - Database Table Creation")
    print("=" * 70)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    print("✓ Database tables created successfully")
    print()
    print("Expected tables:")
    print("  - customers")
    print("  - predictions")
    print("  - model_predictions")
    print("  - risk_factors")
    print("  - model_versions")
    print("=" * 70)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(create_tables())