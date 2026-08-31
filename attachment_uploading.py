"""
附件上传处理：提取上传文件内容并注入到 RAG 检索上下文，与知识库召回结果平级。
"""
import os

# 文件内容最大保留字符数（防止撑爆上下文窗口）
MAX_FILE_CONTENT_LENGTH = 5000

ALLOWED_EXTENSIONS = {"txt", "pdf", "docx", "pptx", "xlsx", "md"}


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否在允许列表中"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


def extract_file_content(file_bytes: bytes, filename: str) -> str:
    """从上传的文件字节中提取文字内容"""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        from knowledge_base import _read_pdf_pages
        pages = _read_pdf_pages(file_bytes)
        return "\n".join(text for _, text in pages)
    else:
        from knowledge_base import read_file_content
        return read_file_content(file_bytes, filename)


def format_file_context(filename: str, content: str) -> str:
    """将文件内容格式化为注入 RAG 上下文的字符串，超长自动截断"""
    if len(content) > MAX_FILE_CONTENT_LENGTH:
        content = content[:MAX_FILE_CONTENT_LENGTH] + "\n...(内容过长，已截断)"

    return f"【用户上传附件 - {filename}】\n{content}\n"
