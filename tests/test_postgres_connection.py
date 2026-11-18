
import asyncio

from app.postgres_db import (
    connect_postgres,
    get_postgres_pool,
    get_farmer_details,
    close_postgres,
)


async def main():
    try:
        print("Connecting to Kisan Dost PostgreSQL...")

        await connect_postgres()

        pool = get_postgres_pool()

        async with pool.acquire() as connection:
            database = await connection.fetchval(
                "SELECT current_database()"
            )
            print(f"Connected successfully: {database}")

        for user_id in [9, 15]:
            print(f"\nFetching farmer ID {user_id}...")

            farmer = await get_farmer_details(user_id)

            if farmer:
                print("Farmer details:")
                print(dict(farmer))
            else:
                print("Farmer not found or inactive.")

    except Exception as error:
        print(f"Database connection failed: {error}")
        raise

    finally:
        await close_postgres()


if __name__ == "__main__":
    asyncio.run(main())
