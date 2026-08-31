"""数据库操作：管理员 + 上传记录 + 文档统计"""
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


def _ensure_tables():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    admin_id   INT AUTO_INCREMENT PRIMARY KEY,
                    username   VARCHAR(50)  NOT NULL UNIQUE,
                    password   VARCHAR(255) NOT NULL,
                    phone      VARCHAR(20)  DEFAULT '',
                    email      VARCHAR(100) DEFAULT '',
                    avatar     MEDIUMTEXT,
                    tags       VARCHAR(500) DEFAULT '',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS file_uploading (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    md5_value     VARCHAR(64) DEFAULT NULL,
                    file_name     VARCHAR(255) NOT NULL,
                    file_type     VARCHAR(20)  NOT NULL,
                    upload_time   DATETIME DEFAULT CURRENT_TIMESTAMP,
                    upload_status VARCHAR(20)  NOT NULL DEFAULT 'success',
                    FOREIGN KEY (md5_value) REFERENCES knowledge_md5(md5_value) ON DELETE SET NULL,
                    INDEX idx_upload_time (upload_time DESC)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
    finally:
        conn.close()


# ==================== 管理员 CRUD ====================

def query_admin_by_username(username: str) -> dict | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM admins WHERE username = %s", (username,))
            return cur.fetchone()
    finally:
        conn.close()


def query_admin_by_id(admin_id: int) -> dict | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM admins WHERE admin_id = %s", (admin_id,))
            return cur.fetchone()
    finally:
        conn.close()


def insert_admin(username: str, password_hash: str, phone: str = "", email: str = "") -> int:
    _ensure_tables()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO admins (username, password, phone, email) VALUES (%s, %s, %s, %s)",
                (username, password_hash, phone, email),
            )
            conn.commit()
            return cur.lastrowid
    finally:
        conn.close()


def update_admin_info(username: str, fields: dict) -> bool:
    allowed = {"phone", "email", "avatar", "tags"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = %s" for k in updates)
    values = list(updates.values()) + [username]
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE admins SET {set_clause} WHERE username = %s", values)
            conn.commit()
            return cur.rowcount > 0
    finally:
        conn.close()


def update_admin_password(username: str, new_password_hash: str) -> bool:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE admins SET password = %s WHERE username = %s",
                (new_password_hash, username),
            )
            conn.commit()
            return cur.rowcount > 0
    finally:
        conn.close()


# ==================== 上传记录 CRUD ====================

def insert_upload_record(md5_value: str | None, file_name: str, file_type: str, upload_status: str):
    _ensure_tables()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO file_uploading (md5_value, file_name, file_type, upload_status) VALUES (%s, %s, %s, %s)",
                (md5_value, file_name, file_type, upload_status),
            )
            conn.commit()
    finally:
        conn.close()


def get_upload_records(limit: int = 50) -> list[dict]:
    _ensure_tables()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, file_name, file_type, upload_time, upload_status FROM file_uploading ORDER BY upload_time DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()


def get_document_stats() -> dict:
    """获取文档统计数据：总数 + 今日上传数 + 按类型计数"""
    _ensure_tables()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT filename) AS total FROM knowledge_md5")
            total = cur.fetchone()["total"]

            cur.execute("""
                SELECT COUNT(*) FROM file_uploading
                WHERE upload_status = 'success' AND DATE(upload_time) = CURDATE()
            """)
            today = cur.fetchone()["COUNT(*)"]

            cur.execute("""
                SELECT LOWER(SUBSTRING_INDEX(filename, '.', -1)) AS file_type, COUNT(DISTINCT filename) AS cnt
                FROM knowledge_md5
                GROUP BY file_type
                ORDER BY cnt DESC
            """)
            by_type = {row["file_type"]: row["cnt"] for row in cur.fetchall()}
        return {"total": total, "today": today, "by_type": by_type}
    finally:
        conn.close()


def get_admin_count() -> int:
    """获取管理员总数"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM admins")
            return cur.fetchone()["cnt"]
    finally:
        conn.close()
