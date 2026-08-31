# ========== 数据库及flask框架配置 ==========
# Redis 配置
redis_host = "127.0.0.1"
redis_port = 6379
redis_db = 0
redis_password = None
chat_history_prefix = "chat_history:"  # Redis key 前缀

# MySQL 配置
mysql_host = "127.0.0.1"
mysql_port = 3306
mysql_user = "root"
mysql_password = "750705"
mysql_database = "电商运营智能助手"
mysql_charset = "utf8mb4"
mysql_pool_min = 2   # 连接池最小连接数
mysql_pool_max = 5   # 连接池最大连接数

# Flask 服务配置
flask_host = "127.0.0.1"
flask_port = 5000
# Flask session 密钥
flask_secret_key = "dianshang-rag-secret-key-2024"

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

# ========== SemanticChunker 语义分割配置 ==========
semantic_breakpoint_threshold_type = "percentile"  # 可选 "percentile", "standard_deviation", "interquartile", "gradient"
semantic_breakpoint_threshold_amount = 90          # 百分位阈值，数值越大切得越碎
max_chunk_api_limit = 2000                          # 单个块最大字符数，防止超过 embedding API 限制（8192）
max_chunk_overlap = 200                             # 单个文本块中允许重叠的最大字符数

# ========== 数据清洗配置 ==========
# 短句碎片最小字符数（清洗后单行不足此长度的将被丢弃）
min_line_char_length = 5

# 目录行匹配模式（连续多行命中则整段视为目录区）
toc_line_patterns = [
    r'\.{4,}',        # 英文省略号引导符
    r'…{2,}',         # 中文省略号引导符
    r'^\s*\d{1,3}\s*$',  # 纯数字页码
    r'[第]?\d+[章节].*\d+\s*$',  # 章节标题后跟页码
    r'^\s*[（(][一二三四五六七八九十百]+[）)]\s*$',  # 中文序号标题
]

# 版权页特征关键词（包含任一关键词的段落将被整段删除）
copyright_keywords = [
    '版权', '版权所有', '翻印必究', '未经许可', '不得复制',
    'ISBN', 'CIP数据', '图书在版编目', '侵权必究', '保留所有权利',
    '著作权', '版次', '印次', '印张', '开本',
]

# 本地模型目录（由 download_model.py 下载）
local_models_dir = "./local_models"

# Reranking 重排序配置
rerank_top_k = 3          # 精排后保留数

# 混合检索配置
hybrid_candidate_k = 6   # 每路召回数（语义 + BM25 各取 N 篇）
hybrid_rrf_k = 60         # RRF 平滑常数
rerank_model_path = local_models_dir + "/bge-reranker-v2-m3"  # 本地路径

# 默认图片检索设置
similarity_threshold = 3

# ========== 分层记忆配置 ==========
# 短期记忆：工作记忆窗口（最近 N 条消息保留原文，超出部分做摘要压缩）
memory_window_size = 6       # 工作记忆窗口大小（6 条 ≈ 3 轮对话）
memory_summary_enabled = True  # 是否启用短期记忆摘要

# ========== 模型配置 ==========
#文本Embedding模型设置
embedding_model_name = "text-embedding-v4"
dashscope_api_key = "sk-cdf4abc8d0124a34b724252c12dd42c9"

#图形Embedding模型设置
image_model_name='clip-ViT-B-32'
image_model_path = local_models_dir + "/clip-ViT-B-32"  # 本地路径

#聊天/短期记忆生成摘要模型设置
chat_model_name = "qwen3.7-max"

#长期记忆主题分类模型设置
topic_classification_model_name = "qwen-flash"

#评估模型设置
#生成测试集问题
generate_question_name = "qwen-flash"
#生成评估数据集参数
num_sample_chunks = 50
questions_per_chunk = 2

#评估模型针对测试问题的回答效果
eval_model_name="qwen-max"
api_key = "sk-cdf4abc8d0124a34b724252c12dd42c9"

