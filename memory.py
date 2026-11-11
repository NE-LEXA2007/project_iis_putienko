"""П5. Сессионная память SQLite; SQL-параметры, изоляция и хронология."""
import sqlite3
from datetime import datetime, timezone

import config
from contracts import Message

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 session_id TEXT NOT NULL,
 role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
 content TEXT NOT NULL,
 timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_session_id ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS session_products (
 session_id TEXT PRIMARY KEY,
 product TEXT NOT NULL CHECK(product IN ('cash','card'))
);
"""


def connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(config.DB_PATH, timeout=20)


def create_schema() -> None:
    with connect() as db:
        db.executescript(SCHEMA)


def save(session_id: str, role: str, content: str) -> None:
    if role not in {"user", "assistant", "system"}:
        raise ValueError("Недопустимая роль сообщения")
    with connect() as db:
        db.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES(?,?,?,?)",
                   (session_id, role, content, datetime.now(timezone.utc).isoformat(timespec="seconds")))


def window(session_id: str, n: int = config.MEMORY_WINDOW) -> list[Message]:
    if n < 0:
        raise ValueError("Размер окна не может быть отрицательным")
    if n == 0:
        return []
    with connect() as db:
        rows = db.execute("SELECT role,content,timestamp FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                          (session_id, n)).fetchall()
    return [Message(*row) for row in reversed(rows)]


def clear(session_id: str) -> None:
    with connect() as db:
        db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        db.execute("DELETE FROM session_products WHERE session_id=?", (session_id,))


def prune(days: int = 30) -> int:
    """Удаление старых сообщений без фоновых задач; запускать по расписанию вручную."""
    if days < 1:
        raise ValueError("Срок хранения должен быть положительным")
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with connect() as db:
        cursor = db.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff,))
        removed = cursor.rowcount
        db.execute("DELETE FROM session_products WHERE session_id NOT IN (SELECT DISTINCT session_id FROM messages)")
        return removed


def get_product(session_id: str) -> str | None:
    with connect() as db:
        row = db.execute("SELECT product FROM session_products WHERE session_id=?", (session_id,)).fetchone()
    return row[0] if row else None


def set_product(session_id: str, product: str) -> None:
    with connect() as db:
        db.execute("INSERT INTO session_products(session_id,product) VALUES(?,?) "
                   "ON CONFLICT(session_id) DO UPDATE SET product=excluded.product", (session_id, product))
