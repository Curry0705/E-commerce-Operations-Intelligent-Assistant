"""
分层记忆管理
- 短期记忆：超出工作记忆窗口的消息自动压缩为摘要（Redis 缓存）
- 长期记忆：用户话题偏好追踪（Redis 存储，LLM 主题分类）
"""
import json
import config_data as config
from file_history_store import _get_redis_client

# ========== 模型懒加载 ==========

_summary_model = None
_topic_model = None


def _get_summary_model():
    """非流式 ChatTongyi，用于短期记忆生成摘要（懒加载）"""
    global _summary_model
    if _summary_model is None:
        from langchain_community.chat_models.tongyi import ChatTongyi
        _summary_model = ChatTongyi(
            api_key=config.api_key,
            model=config.chat_model_name,
            streaming=False,
        )
    return _summary_model


def _get_topic_model():
    """非流式 ChatTongyi，用于长期记忆主题分类（懒加载）"""
    global _topic_model
    if _topic_model is None:
        from langchain_community.chat_models.tongyi import ChatTongyi
        _topic_model = ChatTongyi(
            api_key=config.api_key,
            model= config.topic_classification_model_name,
            streaming=False,
            temperature=0.0,
        )
    return _topic_model


# ========== 短期记忆：对话摘要 ==========

_SUMMARY_PROMPT = (
    "将以下对话历史压缩为简洁的一段摘要，保留关键信息"
    "（用户关注的话题、重要的数据指标、做出的决策、提到的实体和术语）。\n\n"
    "已有的摘要：\n"
    "{existing_summary}\n\n"
    "新的对话内容：\n"
    "{messages_text}\n\n"
    "请输出更新后的完整摘要（不超过300字，只输出摘要本身）："
)


def get_or_update_summary(session_id: str, messages: list) -> str | None:
    """获取当前会话的短期记忆摘要，必要时触发更新

    当消息数 <= 工作记忆窗口时，不需要摘要。
    当超出窗口时，将较早的消息压缩为摘要并缓存到 Redis。
    通过记录已摘要的消息数避免重复生成。
    """
    if not config.memory_summary_enabled:
        return None

    window = config.memory_window_size
    if len(messages) <= window:
        return None

    # 窗口外的消息需要被摘要
    overflow = messages[:-window]
    overflow_count = len(overflow)
    client = _get_redis_client()

    # 检查缓存：已摘要的消息数是否匹配
    cache_idx_key = f"conversation:{session_id}:summary_idx"
    cached_idx = client.get(cache_idx_key)
    if cached_idx and int(cached_idx) == overflow_count:
        cached = client.get(f"conversation:{session_id}:summary")
        if cached:
            return cached

    # 需要重新生成摘要
    existing = client.get(f"conversation:{session_id}:summary") or ""
    summary = _generate_summary(overflow, existing)
    if not summary:
        return existing or None  # 生成失败则使用旧摘要

    client.set(f"conversation:{session_id}:summary", summary)
    client.set(cache_idx_key, str(overflow_count))
    return summary


def _generate_summary(messages: list, existing_summary: str) -> str:
    """用 LLM 将消息列表压缩为一段摘要"""
    lines = []
    for msg in messages:
        role = "用户" if msg.type == "human" else "助手"
        lines.append(f"[{role}]: {msg.content}")
    messages_text = "\n".join(lines)

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("user", _SUMMARY_PROMPT),
        ])
        chain = prompt | _get_summary_model() | StrOutputParser()
        result = chain.invoke({
            "existing_summary": existing_summary or "无",
            "messages_text": messages_text,
        })
        if result and result.strip():
            return result.strip()
    except Exception:
        pass
    return ""


def clear_summary(session_id: str):
    """清除会话的摘要缓存"""
    client = _get_redis_client()
    client.delete(f"conversation:{session_id}:summary")
    client.delete(f"conversation:{session_id}:summary_idx")
    client.delete(f"conversation:{session_id}:stopped")
    client.delete(f"conversation:{session_id}:stopped_partial")


def decrement_query_topics(user_id: str, session_id: str, history=None):
    """删除/清空会话时，从会话级 topic 记录回退用户画像计数"""
    if not user_id:
        return

    client = _get_redis_client()
    session_key = f"session:{session_id}:topic_counts"
    session_counts = client.hgetall(session_key)

    if not session_counts:
        return

    hash_key = f"user_profile:{user_id}:topics"
    total_to_decr = 0
    for topic, count_str in session_counts.items():
        count = int(count_str)
        if topic == "_total_queries":
            total_to_decr = count
            continue
        current = int(client.hget(hash_key, topic) or 0)
        if current > 0:
            decr = min(count, current)
            if decr > 0:
                client.hincrby(hash_key, topic, -decr)

    if total_to_decr > 0:
        current_total = int(client.hget(hash_key, "_total_queries") or 0)
        if current_total > 0:
            client.hincrby(hash_key, "_total_queries", -min(total_to_decr, current_total))

    # 清理会话级记录
    client.delete(session_key)


# ========== 长期记忆：用户话题偏好 ==========

TOPIC_CLASSIFY_PROMPT = """你是一个电商运营主题分类器。根据用户问题，判断属于以下哪些主题：

引流推广、付费推广、转化成交、商品管理、视觉设计、定价策略、
店铺运营、平台规则、数据分析、仓储物流、采购供应、活动促销、
内容营销、直播运营、短视频、客服售后、会员营销、竞品分析、人群定位、其他

规则：
1. 只从上述 20 个主题中选择，不要自创主题
2. 可以多选，无关则返回空数组
3. 相近问题合并为同主题，如"直通车"和"钻展"都归"付费推广"
4. 仅返回 JSON 数组，不要输出任何其他内容

示例：
"直通车ROI怎么提升" → ["付费推广", "数据分析"]
"618大促活动怎么报名" → ["活动促销"]
"淘宝店铺怎么装修" → ["店铺运营", "视觉设计"]
"直播带货话术怎么设计" → ["直播运营", "内容营销"]
"今天天气怎么样" → [其他]

用户问题：{query}
分类结果："""


def _classify_topics(query: str) -> list[str]:
    """用 LLM 对 query 做主题分类，返回主题标签列表"""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([("user", TOPIC_CLASSIFY_PROMPT)])
        chain = prompt | _get_topic_model()
        response = chain.invoke({"query": query})
        text = response.content.strip()
        # 清理可能的 markdown 代码块包裹
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if lines[0].startswith("```") else text
            if text.endswith("```"):
                text = text[:-3]
        topics = json.loads(text)
        if isinstance(topics, list):
            return [t for t in topics if isinstance(t, str)]
    except Exception:
        pass
    return []


def track_query_topics(user_id: str, session_id: str, query: str):
    """LLM 主题分类 → 更新用户话题频次，同时记录会话级主题用于回退"""
    if not query:
        return

    topics = _classify_topics(query)
    if not topics:
        return

    client = _get_redis_client()
    hash_key = f"user_profile:{user_id}:topics"

    for topic in topics:
        client.hincrby(hash_key, topic, 1)
        # 会话级记录，删除会话时用于递减
        client.hincrby(f"session:{session_id}:topic_counts", topic, 1)

    client.hincrby(hash_key, "_total_queries", 1)
    client.hincrby(f"session:{session_id}:topic_counts", "_total_queries", 1)


def get_user_profile(user_id: str) -> dict:
    """获取用户长期偏好画像"""
    client = _get_redis_client()
    hash_key = f"user_profile:{user_id}:topics"
    raw = client.hgetall(hash_key)

    if not raw:
        return {"topics": {}, "total_queries": 0}

    total = int(raw.pop("_total_queries", 0))
    topics = {k: int(v) for k, v in sorted(
        raw.items(), key=lambda x: int(x[1]), reverse=True
    )}
    return {"topics": topics, "total_queries": total}


def clear_user_profile(user_id: str):
    """清除用户画像"""
    client = _get_redis_client()
    client.delete(f"user_profile:{user_id}:topics")
