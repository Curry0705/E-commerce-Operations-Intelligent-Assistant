"""
一键下载项目所需的所有本地模型（仅需运行一次）

使用方式: python download_model.py

下载后模型存放于 ./local_models/ 目录，项目代码直接读取本地路径。
"""
import os
import sys
import shutil

LOCAL_DIR = os.path.join(os.path.dirname(__file__), "local_models")

# 模型清单: { 本地目录名: (ModelScope ID, HF ID) }
MODELS = {
    "clip-ViT-B-32": {
        "ms_id": "iic/clip-vit-base-patch32",
        "hf_id": "sentence-transformers/clip-ViT-B-32",
        "type": "sentence_transformer",
    },
    "bge-reranker-v2-m3": {
        "ms_id": "BAAI/bge-reranker-v2-m3",
        "hf_id": "BAAI/bge-reranker-v2-m3",
        "type": "cross_encoder",
    },
    "table-transformer-detection": {
        "ms_id": None,
        "hf_id": "microsoft/table-transformer-detection",
        "type": "transformers",
    },
    "table-transformer-structure-recognition": {
        "ms_id": None,
        "hf_id": "microsoft/table-transformer-structure-recognition",
        "type": "transformers",
    },
}

# 多个 HF 镜像端点, 依次尝试
HF_MIRRORS = [
    "https://hf-mirror.com",
    "https://hf.xeduapi.com",
    None,  # None = 直连 huggingface.co
]


def _try_ms_download(ms_id: str, target_dir: str) -> bool:
    """通过 ModelScope 下载模型, 成功返回 True"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        downloaded = snapshot_download(ms_id, cache_dir=target_dir)
        if downloaded and os.path.isdir(downloaded):
            # snapshot_download 会在 target_dir 下创建子目录, 移动文件到 target_dir
            if downloaded != target_dir:
                for item in os.listdir(downloaded):
                    s = os.path.join(downloaded, item)
                    d = os.path.join(target_dir, item)
                    if os.path.isdir(s):
                        if os.path.exists(d):
                            shutil.rmtree(d)
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)
            return True
    except Exception as e:
        print(f"    ModelScope 下载失败: {e}")
    return False


def _try_hf_download(hf_id: str, target_dir: str) -> bool:
    """依次尝试多个 HF 镜像下载模型"""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    for mirror in HF_MIRRORS:
        if mirror:
            os.environ["HF_ENDPOINT"] = mirror
        else:
            os.environ.pop("HF_ENDPOINT", None)
        try:
            # 下载到 HF 缓存
            proc = AutoImageProcessor.from_pretrained(hf_id)
            model = AutoModelForObjectDetection.from_pretrained(hf_id)
            # 下载成功, 保存到 target_dir
            proc.save_pretrained(target_dir)
            model.save_pretrained(target_dir)
            return True
        except Exception as e:
            print(f"    端点 {mirror or 'huggingface.co'}: {e}")
    return False


def _try_hf_sentence_download(hf_id: str, target_dir: str) -> bool:
    """通过 HF 下载 SentenceTransformer 模型"""
    from sentence_transformers import SentenceTransformer

    for mirror in HF_MIRRORS:
        if mirror:
            os.environ["HF_ENDPOINT"] = mirror
        else:
            os.environ.pop("HF_ENDPOINT", None)
        try:
            model = SentenceTransformer(hf_id)
            model.save(target_dir)
            return True
        except Exception as e:
            print(f"    端点 {mirror or 'huggingface.co'}: {e}")
    return False


def _try_hf_crossencoder_download(hf_id: str, target_dir: str) -> bool:
    """通过 HF 下载 CrossEncoder 模型"""
    from sentence_transformers import CrossEncoder

    for mirror in HF_MIRRORS:
        if mirror:
            os.environ["HF_ENDPOINT"] = mirror
        else:
            os.environ.pop("HF_ENDPOINT", None)
        try:
            model = CrossEncoder(hf_id)
            model.save(target_dir)
            return True
        except Exception as e:
            print(f"    端点 {mirror or 'huggingface.co'}: {e}")
    return False


def download():
    # 确保不在离线模式
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.makedirs(LOCAL_DIR, exist_ok=True)

    success_count = 0
    fail_count = 0

    for name, info in MODELS.items():
        target_dir = os.path.join(LOCAL_DIR, name)
        print(f"\n{'='*50}")
        print(f"  [{info['type']}] {name}")
        print(f"{'='*50}")

        # 已存在 → 跳过
        if os.path.isdir(target_dir) and os.listdir(target_dir):
            print(f"  ✓ 已存在, 跳过")
            success_count += 1
            continue

        os.makedirs(target_dir, exist_ok=True)
        ok = False

        # 策略1: ModelScope
        if info["ms_id"]:
            print(f"  尝试 ModelScope: {info['ms_id']}")
            ok = _try_ms_download(info["ms_id"], target_dir)

        # 策略2: HF 多镜像
        if not ok:
            print(f"  尝试 HF 镜像: {info['hf_id']}")
            if info["type"] == "sentence_transformer":
                ok = _try_hf_sentence_download(info["hf_id"], target_dir)
            elif info["type"] == "cross_encoder":
                ok = _try_hf_crossencoder_download(info["hf_id"], target_dir)
            else:
                ok = _try_hf_download(info["hf_id"], target_dir)

        if ok:
            print("  ✓ 完成")
            success_count += 1
        else:
            print(f"  ✗ 失败: 所有下载方式均不可用")
            fail_count += 1

    # ========== PaddleOCR (单独处理) ==========
    print(f"\n{'='*50}")
    print("  [paddleocr] PaddleOCR 中文模型")
    print(f"{'='*50}")
    try:
        from paddleocr import PaddleOCR
        PaddleOCR(lang='ch')
        print("  ✓ 完成 (模型已缓存)")
        success_count += 1
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        fail_count += 1

    # ========== 结果汇总 ==========
    total = len(MODELS) + 1
    print(f"\n{'='*50}")
    print(f"  下载完成: 成功 {success_count}/{total}, 失败 {fail_count}/{total}")
    print(f"{'='*50}")

    if fail_count > 0:
        print("\n部分模型下载失败, 请检查网络连接或手动下载。")
        sys.exit(1)
    else:
        print(f"\n所有模型已下载到 {LOCAL_DIR}")
        print("运行 pip install timm 后再启动项目。")
        print("\n可运行 python download_model.py 查看状态。")


if __name__ == "__main__":
    download()
