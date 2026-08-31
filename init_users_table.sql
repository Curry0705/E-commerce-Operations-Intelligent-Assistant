-- ============================================
-- 电商运营智能助手 — 数据库初始化脚本
-- ============================================

CREATE DATABASE IF NOT EXISTS `电商运营智能助手`
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE `电商运营智能助手`;

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    user_id    INT AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(50)  NOT NULL UNIQUE,
    password   VARCHAR(255) NOT NULL,
    phone      VARCHAR(20)  DEFAULT '',
    email      VARCHAR(100) DEFAULT '',
    avatar     MEDIUMTEXT,
    tags       VARCHAR(500) DEFAULT '',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 知识库文件 MD5 去重表
CREATE TABLE IF NOT EXISTS knowledge_md5 (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    md5_value   VARCHAR(64) NOT NULL UNIQUE,
    filename    VARCHAR(255) DEFAULT '',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_filename (filename)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 知识库管理员表
CREATE TABLE IF NOT EXISTS admins (
    admin_id   INT AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(50)  NOT NULL UNIQUE,
    password   VARCHAR(255) NOT NULL,
    phone      VARCHAR(20)  DEFAULT '',
    email      VARCHAR(100) DEFAULT '',
    avatar     MEDIUMTEXT,
    tags       VARCHAR(500) DEFAULT '',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 知识库文件上传记录表
CREATE TABLE IF NOT EXISTS file_uploading (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    md5_value     VARCHAR(64) DEFAULT NULL,
    file_name     VARCHAR(255) NOT NULL,
    file_type     VARCHAR(20)  NOT NULL,
    upload_time   DATETIME DEFAULT CURRENT_TIMESTAMP,
    upload_status VARCHAR(20)  NOT NULL DEFAULT 'success',
    FOREIGN KEY (md5_value) REFERENCES knowledge_md5(md5_value) ON DELETE SET NULL,
    INDEX idx_upload_time (upload_time DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

