
import os
from uuid import uuid4, UUID

import asyncpg
from dotenv import load_dotenv

load_dotenv(".env")


class ConversationStore:
    def __init__(self, settings=None):
        self.pool = None

    async def connect(self):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                host=os.environ["POSTGRES_HOST"],
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.environ["POSTGRES_DATABASE"],
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
                min_size=1,
                max_size=10,
            )

    async def close(self):
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    @staticmethod
    def new_session_id() -> str:
        return str(uuid4())

    async def save_turn(
        self,
        session_id: str,
        language: str,
        user_message: str,
        assistant_message: str,
        intent: str,
    ) -> None:
        await self.connect()

        await self.pool.execute(
            """
            INSERT INTO public.voice_conversation_turns
            (session_id, language, user_message, assistant_message, intent)
            VALUES ($1, $2, $3, $4, $5)
            """,
            UUID(session_id),
            language,
            user_message,
            assistant_message,
            intent,
        )

    async def get_history(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        await self.connect()

        rows = await self.pool.fetch(
            """
            SELECT user_message, assistant_message
            FROM public.voice_conversation_turns
            WHERE session_id = $1
            ORDER BY created_at DESC, id DESC
            LIMIT $2
            """,
            UUID(session_id),
            limit,
        )

        history = []
        for row in reversed(rows):
            history.append({"role": "user", "content": row["user_message"]})
            history.append(
                {"role": "assistant", "content": row["assistant_message"]}
            )

        return history
