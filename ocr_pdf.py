"""
OCR 处理扫描件 PDF，将页面渲染为图片后通过 PaddleOCR 提取文字
"""
import io
import numpy as np
import fitz  # PyMuPDF
from PIL import Image


_ocr = None


def _get_ocr():
    """延迟加载 PaddleOCR 单例（首次调用时自动下载模型）"""
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(lang='ch')
    return _ocr


def _bytes_to_array(png_bytes: bytes) -> np.ndarray:
    """将 PNG 字节数据转为 PaddleOCR 可接受的 numpy 数组"""
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))


def ocr_pdf(file_bytes: bytes, dpi: int = 150) -> str:
    """对 PDF 逐页 OCR，返回提取的文字

    Args:
        file_bytes: PDF 文件字节数据
        dpi: 渲染分辨率，越大 OCR 越准但越慢（推荐 200）

    Returns:
        所有页面 OCR 结果的合并文本，页间用换行分隔
    """
    import sys

    ocr = _get_ocr()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total = len(doc)
    text_parts = []

    for page_num in range(total):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img_array = _bytes_to_array(pix.tobytes("png"))

        result = list(ocr.ocr(img_array))
        page_text = ""
        if result and result[0]:
            rec_texts = result[0].get("rec_texts", [])
            if rec_texts:
                page_text = "\n".join(rec_texts)
                text_parts.append(page_text)

        sys.stderr.write(f"\r[OCR] {page_num + 1}/{total} 页   ")
        sys.stderr.flush()

    doc.close()
    print(file=sys.stderr)  # 换行
    return "\n".join(text_parts)


def ocr_pdf_to_images(file_bytes: bytes, dpi: int = 150) -> list[dict]:
    """对 PDF 逐页 OCR，返回每页的图片+文字结构化数据

    Args:
        file_bytes: PDF 文件字节数据
        dpi: 渲染分辨率

    Returns:
        [{"page_num": 1, "text": "...", "image_bytes": b"..."}, ...]
    """
    ocr = _get_ocr()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        img_array = _bytes_to_array(img_bytes)

        result = list(ocr.ocr(img_array))
        page_text = ''
        if result and result[0]:
            rec_texts = result[0].get("rec_texts", [])
            if rec_texts:
                page_text = '\n'.join(rec_texts)

        results.append({
            "page_num": page_num + 1,
            "text": page_text,
            "image_bytes": img_bytes,
        })

    doc.close()
    return results
