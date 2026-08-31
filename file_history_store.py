import json
import traceback
from datetime import datetime
from typing import Sequence, Optional
import redis
import config_data as config
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

_redis_client: Optional[redis.Redis] = None


def _get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            decode_responses=True,
        )
        _redis_client.ping()
    return _redis_client


def get_history(session_id: str):
    return RedisChatMessageHistory(session_id)


def get_all_conversations(user_id: int = None) -> list[dict]:
    """返回当前用户的会话列表，按时间倒序"""
    client = _get_redis_client()
    if user_id:
        ids = client.lrange(f"user:{user_id}:conversations", 0, -1)
    else:
        ids = client.lrange("conversations:list", 0, -1)
    result = []
    for cid in ids:
        meta_raw = client.get(f"conversation:{cid}:meta")
        if meta_raw:
            meta = json.loads(meta_raw)
            result.append({"id": cid, "title": meta.get("title", ""), "created_at": meta.get("created_at", "")})
    return result


def save_conversation_meta(session_id: str, title: str, user_id: int = None):
    """保存或更新会话元信息，并关联到用户"""
    client = _get_redis_client()
    meta = {"title": title, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user_id": user_id}
    client.set(f"conversation:{session_id}:meta", json.dumps(meta, ensure_ascii=False))
    client.lrem("conversations:list", 0, session_id)
    client.lpush("conversations:list", session_id)
    # 关联到用户
    if user_id:
        client.lrem(f"user:{user_id}:conversations", 0, session_id)
        client.lpush(f"user:{user_id}:conversations", session_id)


def delete_conversation(session_id: str):
    """删除会话及其元信息"""
    client = _get_redis_client()
    # 获取会话元信息以得到 user_id
    meta_raw = client.get(f"conversation:{session_id}:meta")
    user_id = None
    if meta_raw:
        meta = json.loads(meta_raw)
        user_id = meta.get("user_id")
    client.delete(f"{config.chat_history_prefix}{session_id}")
    client.delete(f"conversation:{session_id}:meta")
    client.delete(f"conversation:{session_id}:summary")
    client.delete(f"conversation:{session_id}:summary_idx")
    client.delete(f"conversation:{session_id}:stopped")
    client.delete(f"conversation:{session_id}:stopped_partial")
    client.delete(f"conversation:{session_id}:stopped_user_msg")
    client.delete(f"conversation:{session_id}:stopped_snapshot")
    client.lrem("conversations:list", 0, session_id)
    if user_id:
        client.lrem(f"user:{user_id}:conversations", 0, session_id)


class RedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.redis_key = f"{config.chat_history_prefix}{session_id}"

    @property
    def client(self) -> redis.Redis:
        return _get_redis_client()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        try:
            all_messages = list(self.messages)
            all_messages.extend(messages)
            new_messages = [message_to_dict(message) for message in all_messages]
            self.client.set(self.redis_key, json.dumps(new_messages, ensure_ascii=False))
        except Exception:
            traceback.print_exc()

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            data = self.client.get(self.redis_key)
            if data is None:
                return []
            messages_data = json.loads(data)
            return messages_from_dict(messages_data)
        except Exception:
            traceback.print_exc()
            return []

    def clear(self) -> None:
        try:
            self.client.delete(self.redis_key)
        except Exception:
            traceback.print_exc()
