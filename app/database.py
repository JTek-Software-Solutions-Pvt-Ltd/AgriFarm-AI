import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4
from app.config import Settings


class ConversationStore:
    def __init__(self, settings: Settings):
        self.path = settings.database_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                language TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL,
                intent TEXT NOT NULL
            )""")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def new_session_id() -> str:
        return str(uuid4())

    def save_turn(self, session_id: str, language: str, user_message: str, assistant_message: str, intent: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO turns(session_id, created_at, language, user_message, assistant_message, intent) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, datetime.now(timezone.utc).isoformat(), language, user_message, assistant_message, intent),
            )
