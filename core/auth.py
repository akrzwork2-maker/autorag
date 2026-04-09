from __future__ import annotations

import os
import sqlite3
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import jwt

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "users.db"
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

_conn: sqlite3.Connection | None = None
_enabled = False


async def init_db():
    """Initialise local SQLite user database."""
    global _conn, _enabled
    try:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        _conn.commit()
        _enabled = True
        logger.info("Auth database ready — %s", _DB_PATH)
        return True
    except Exception as e:
        logger.warning("Auth database init failed: %s", e)
        _conn = None
        _enabled = False
        return False


async def close_db():
    global _conn, _enabled
    if _conn:
        _conn.close()
        _conn = None
    _enabled = False


def is_auth_enabled() -> bool:
    return _enabled


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(user_id: str, email: str, name: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


async def create_user(name: str, email: str, password: str) -> dict:
    if not is_auth_enabled():
        raise RuntimeError("Auth not configured")

    email = email.strip().lower()
    row = _conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if row:
        raise ValueError("Email already registered")

    user_id = uuid.uuid4().hex
    _conn.execute(
        "INSERT INTO users (id, name, email, password, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, name.strip(), email, hash_password(password), datetime.now(timezone.utc).isoformat()),
    )
    _conn.commit()
    return {"id": user_id, "name": name.strip(), "email": email}


async def authenticate_user(email: str, password: str) -> dict | None:
    if not is_auth_enabled():
        raise RuntimeError("Auth not configured")

    email = email.strip().lower()
    row = _conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not row:
        return None
    if not verify_password(password, row["password"]):
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


async def get_user_by_id(user_id: str) -> dict | None:
    if not is_auth_enabled():
        return None
    row = _conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"]}
