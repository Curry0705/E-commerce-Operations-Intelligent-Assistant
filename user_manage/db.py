"""数据库连接池 + 查询封装"""
import pymysql
from dbutils.pooled_db import PooledDB
import config_data as config

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = PooledDB(
            creator=pymysql,
            mincached=config.mysql_pool_min,
            maxconnections=config.mysql_pool_max,
            host=config.mysql_host,
            port=config.mysql_port,
            user=config.mysql_user,
            password=config.mysql_password,
            database=config.mysql_database,
            charset=config.mysql_charset,
            cursorclass=pymysql.cursors.DictCursor,
        )
    return _pool


def get_conn():
    return _get_pool().connection()


def query_user_by_username(username: str) -> dict | None:
    """按用户名查询用户"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            return cur.fetchone()
    finally:
        conn.close()


def query_user_by_id(user_id: int) -> dict | None:
    """按用户 ID 查询用户"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            return cur.fetchone()
    finally:
        conn.close()


def insert_user(username: str, password_hash: str, phone: str = "", email: str = "") -> int:
    """插入新用户，返回 user_id"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password, phone, email) VALUES (%s, %s, %s, %s)",
                (username, password_hash, phone, email),
            )
            conn.commit()
            return cur.lastrowid
    finally:
        conn.close()


def update_user_info(username: str, fields: dict) -> bool:
    """更新用户信息（phone, email, avatar, tags），返回是否成功"""
    allowed = {"phone", "email", "avatar", "tags"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False

    set_clause = ", ".join(f"{k} = %s" for k in updates)
    values = list(updates.values()) + [username]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE users SET {set_clause} WHERE username = %s", values)
            conn.commit()
            return cur.rowcount > 0
    finally:
        conn.close()


def update_password(username: str, new_password_hash: str) -> bool:
    """更新用户密码"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password = %s WHERE username = %s",
                (new_password_hash, username),
            )
            conn.commit()
            return cur.rowcount > 0
    finally:
        conn.close()
