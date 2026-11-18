import os

from dotenv import load_dotenv
import asyncpg
from asyncpg import Pool

load_dotenv()

_pool: Pool | None = None


async def connect_postgres() -> None:
    """Create a shared PostgreSQL connection pool."""
    global _pool
    if _pool is not None:
        return

    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError(
            "POSTGRES_PASSWORD is missing. Check the project .env file."
        )

    _pool = await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        database=os.getenv("POSTGRES_DATABASE", "kisandost"),
        user=os.getenv("POSTGRES_USER", "kdadmin"),
        password=password,
        min_size=1,
        max_size=5,
        command_timeout=30,
    )


async def close_postgres() -> None:
    """Close the PostgreSQL connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_postgres_pool() -> Pool:
    """Return the initialized PostgreSQL pool."""
    if _pool is None:
        raise RuntimeError("PostgreSQL connection is not initialized.")
    return _pool


async def get_farmer_details(user_id: int):
    """
    Fetch farmer and selected crop information using only existing columns.
    This query does not modify the database.
    """
    pool = get_postgres_pool()
    query = """
        SELECT
            u.id AS farmer_id,
            u.crop_id,
            u.country,
            u.state,
            u.district,
            u.village,
            u.pincode,
            u.latitude,
            u.longitude,
            c.name AS crop_name,
            c.scientific_name,
            c.category,
            c.season,
            c.crop_details
        FROM public.users u
        LEFT JOIN public.crop c ON c.id = u.crop_id
        WHERE u.id = $1 AND u.is_active = TRUE
    """
    async with pool.acquire() as connection:
        return await connection.fetchrow(query, user_id)


async def save_farmer_coordinates(
    user_id: int,
    latitude: float,
    longitude: float,
    source: str = "open_meteo_geocoding",
) -> None:
    """
    Compatibility function for the weather route.

    Intentionally performs no database write because the user requested
    that the database remain completely unchanged.
    """
    return None
