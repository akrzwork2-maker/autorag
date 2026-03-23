from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGODB_URI", "")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24
DB_NAME = "verifai"

_client: AsyncIOMotorClient | None = None
_db = None


async def init_db():
    global _client, _db
    if not MONGO_URI:
        logger.warning("MONGODB_URI not set — auth disabled")
        return False
    _client = AsyncIOMotorClient(MONGO_URI)
    _db = _client[DB_NAME]
    await _db.users.create_index("email", unique=True)
    logger.info("MongoDB connected — database: %s", DB_NAME)
    return True


async def close_db():
    global _client
    if _client:
        _client.close()
        _client = None


def is_auth_enabled() -> bool:
    return _db is not None


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
    existing = await _db.users.find_one({"email": email})
    if existing:
        raise ValueError("Email already registered")

    doc = {
        "name": name.strip(),
        "email": email,
        "password": hash_password(password),
        "created_at": datetime.now(timezone.utc),
    }
    result = await _db.users.insert_one(doc)
    return {"id": str(result.inserted_id), "name": doc["name"], "email": doc["email"]}


async def authenticate_user(email: str, password: str) -> dict | None:
    if not is_auth_enabled():
        raise RuntimeError("Auth not configured")

    email = email.strip().lower()
    user = await _db.users.find_one({"email": email})
    if not user:
        return None
    if not verify_password(password, user["password"]):
        return None
    return {"id": str(user["_id"]), "name": user["name"], "email": user["email"]}


async def get_user_by_id(user_id: str) -> dict | None:
    if not is_auth_enabled():
        return None
    from bson import ObjectId
    user = await _db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        return None
    return {"id": str(user["_id"]), "name": user["name"], "email": user["email"]}
