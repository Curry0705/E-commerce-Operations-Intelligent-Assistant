"""
基于 Flask + Vue 3 完成电商运营智能助手 QA 服务

启动方式: python app_qa.py
访问地址: http://127.0.0.1:5001
知识库管理: http://127.0.0.1:5000
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"                        # 模型已缓存到本地，禁止联网下载
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"  # 禁用 MKLDNN，修复 PIR 属性兼容问题
os.environ["GLOG_minloglevel"] = "2"                         # 屏蔽 PaddleOCR C++ 底层调试日志

import json
import threading

from flask import Flask, request, jsonify, Response, send_from_directory, session, redirect
from rag import RagService
from file_history_store import get_history, get_all_conversations, save_conversation_meta, delete_conversation, _get_redis_client
import memory_manager
import config_data as config
from attachment_uploading import allowed_file, extract_file_content, format_file_context
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = config.flask_secret_key

# 注册认证
from user_manage.login import login_bp
from user_manage.register import register_bp
from user_manage.alter_password import alter_password_bp
from user_manage.user_information import user_information_bp
app.register_blueprint(login_bp)
app.register_blueprint(register_bp)
app.register_blueprint(alter_password_bp)
app.register_blueprint(user_information_bp)

rag_service = RagService()

# 用于取消正在进行的 LLM 生成
_cancel_events: dict[str, threading.Event] = {}


def _start_kb_service():
    """在后台线程启动知识库管理服务 (端口 5000)"""
    from knowledge_manage.kb_app import app as kb_app
    import sys
    sys.stderr.write("[app_qa] 知识库服务启动中...\n")
    sys.stderr.flush()
    kb_app.run(host=config.flask_host, port=config.flask_port, debug=False, use_reloader=False)


@app.before_request
def check_auth():
    """除公开路径外，其他页面和 API 均需登录"""
    path = request.path

    # 公开路径直接放行
    if path in ("/login.html", "/register.html", "/alter_password.html") \
            or path.startswith("/api/auth/") \
            or path.startswith("/static/"):
        return None

    # 未登录 → 全部拦截
    if "user_id" not in session:
        if path.startswith("/api/"):
            return jsonify({"success": False, "message": "未登录", "redirect": "/login.html"}), 401
        return redirect("/login.html")

    return None


@app.route("/")
def qa_page():
    return send_from_directory("static/users", "qa.html")


@app.route("/login.html")
def login_page():
    return send_from_directory("static/users", "login.html")


@app.route("/register.html")
def register_page():
    return send_from_directory("static/users", "register.html")


@app.route("/alter_password.html")
def alter_password_page():
    return send_from_directory("static/users", "alter_password.html")


@app.route("/user_information.html")
def user_information_page():
    return send_from_directory("static/users", "user_information.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """流式对话 API (SSE)"""
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"success": False, "message": "请输入消息"}), 400

    user_message = data["message"]
    session_id = data.get("session_id", "user_001")

    # 清除该会话的终止标记和快照（新消息来了，之前的终止已失效）
    client = _get_redis_client()
    client.delete(f"conversation:{session_id}:stopped")
    client.delete(f"conversation:{session_id}:stopped_snapshot")

    session_config = {
        "configurable": {
            "session_id": session_id,
        }
    }

    import queue
    import threading

    chunk_queue = queue.Queue()
    cancel_event = threading.Event()
    _cancel_events[session_id] = cancel_event
    user_id = session.get("user_id")

    def run_chain():
        with app.app_context():
            try:
                res_stream = rag_service.chain.stream(
                    {"input": user_message, "session_id": session_id, "user_id": user_id}, session_config
                )
                for chunk in res_stream:
                    if cancel_event.is_set():
                        chunk_queue.put(('cancelled', None))
                        return
                    if chunk:
                        chunk_queue.put(('chunk', chunk))
                chunk_queue.put(('done', None))
            except Exception as e:
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                chunk_queue.put(('error', str(e)))

    t = threading.Thread(target=run_chain, daemon=True)
    t.start()

    def generate():
        try:
            while True:
                item = chunk_queue.get()
                kind, value = item
                if kind == 'chunk':
                    yield f"data: {json.dumps({'chunk': value}, ensure_ascii=False)}\n\n"
                elif kind == 'done':
                    yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
                    break
                elif kind == 'cancelled':
                    yield f"data: {json.dumps({'cancelled': True}, ensure_ascii=False)}\n\n"
                    break
                elif kind == 'error':
                    yield f"data: {json.dumps({'error': value}, ensure_ascii=False)}\n\n"
                    break
        finally:
            cancel_event.set()
            _cancel_events.pop(session_id, None)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/chat/upload", methods=["POST"])
def api_chat_upload():
    """带附件上传的流式对话 API (SSE)"""
    user_message = request.form.get("message", "")
    session_id = request.form.get("session_id", "user_001")

    if not user_message:
        return jsonify({"success": False, "message": "请输入消息"}), 400

    file_context = ""
    attached_filename = ""
    file = request.files.get("file")
    if file and file.filename:
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": f"不支持的文件格式，仅支持: txt, pdf, docx, pptx, xlsx"
            }), 400
        attached_filename = file.filename
        try:
            file_bytes = file.read()
            content = extract_file_content(file_bytes, file.filename)
            file_context = format_file_context(file.filename, content)
        except Exception as e:
            return jsonify({"success": False, "message": f"文件处理失败: {str(e)}"}), 500

    # 清除该会话的终止标记
    client = _get_redis_client()
    client.delete(f"conversation:{session_id}:stopped")

    session_config = {
        "configurable": {
            "session_id": session_id,
        }
    }

    import queue
    import threading

    chunk_queue = queue.Queue()
    cancel_event = threading.Event()
    _cancel_events[session_id] = cancel_event
    user_id = session.get("user_id")

    def run_chain():
        with app.app_context():
            try:
                chain_input = {
                    "input": user_message,
                    "session_id": session_id,
                    "user_id": user_id,
                }
                if file_context:
                    chain_input["file_context"] = file_context

                res_stream = rag_service.chain.stream(chain_input, session_config)
                for chunk in res_stream:
                    if cancel_event.is_set():
                        chunk_queue.put(('cancelled', None))
                        return
                    if chunk:
                        chunk_queue.put(('chunk', chunk))
                chunk_queue.put(('done', None))
            except Exception as e:
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                chunk_queue.put(('error', str(e)))

    t = threading.Thread(target=run_chain, daemon=True)
    t.start()

    def generate():
        try:
            while True:
                item = chunk_queue.get()
                kind, value = item
                if kind == 'chunk':
                    yield f"data: {json.dumps({'chunk': value}, ensure_ascii=False)}\n\n"
                elif kind == 'done':
                    yield f"data: {json.dumps({'done': True, 'filename': attached_filename}, ensure_ascii=False)}\n\n"
                    break
                elif kind == 'cancelled':
                    yield f"data: {json.dumps({'cancelled': True}, ensure_ascii=False)}\n\n"
                    break
                elif kind == 'error':
                    yield f"data: {json.dumps({'error': value}, ensure_ascii=False)}\n\n"
                    break
        finally:
            cancel_event.set()
            _cancel_events.pop(session_id, None)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """终止正在进行的 LLM 生成，保存完整对话快照"""
    data = request.get_json() or {}
    session_id = data.get("session_id", "user_001")
    partial_content = data.get("partial_content", "")
    user_message = data.get("user_message", "")
    cancel_event = _cancel_events.get(session_id)
    if cancel_event:
        cancel_event.set()
    client = _get_redis_client()
    client.set(f"conversation:{session_id}:stopped", "1")
    # 保存完整对话快照：历史消息 + 用户问题 + 部分回答/终止标记
    history = get_history(session_id)
    snapshot = []
    for msg in history.messages:
        snapshot.append({"role": msg.type, "content": msg.content})
    if user_message:
        # 避免重复：检查用户消息是否已在历史中
        already = any(m["role"] == "human" and m["content"] == user_message for m in snapshot)
        if not already:
            snapshot.append({"role": "human", "content": user_message})
    if partial_content:
        snapshot.append({"role": "ai", "content": partial_content, "stopped": True})
    else:
        snapshot.append({"role": "ai", "content": "已终止回答", "stopped": True})
    client.set(f"conversation:{session_id}:stopped_snapshot", json.dumps(snapshot, ensure_ascii=False))
    return jsonify({"success": True})


@app.route("/api/history", methods=["GET"])
def api_history():
    """获取对话历史"""
    session_id = request.args.get("session_id", "user_001")
    history = get_history(session_id)
    messages = []
    for msg in history.messages:
        messages.append({
            "role": msg.type,
            "content": msg.content,
        })
    # 如果该会话被终止过，使用完整快照保证历史完整
    client = _get_redis_client()
    if client.get(f"conversation:{session_id}:stopped"):
        snapshot_raw = client.get(f"conversation:{session_id}:stopped_snapshot")
        if snapshot_raw:
            return jsonify({"success": True, "messages": json.loads(snapshot_raw)})
    return jsonify({"success": True, "messages": messages})


@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    """清空对话历史"""
    data = request.get_json() or {}
    session_id = data.get("session_id", "user_001")
    history = get_history(session_id)
    # 先递减话题偏好（此时 history 中还有消息）
    user_id = session.get("user_id")
    if user_id:
        memory_manager.decrement_query_topics(str(user_id), session_id, history)
    # 再清空
    history.clear()
    memory_manager.clear_summary(session_id)
    # 清空对话时同步清理终止标记
    client = _get_redis_client()
    client.delete(f"conversation:{session_id}:stopped")
    client.delete(f"conversation:{session_id}:stopped_partial")
    return jsonify({"success": True, "message": "对话历史已清空"})


@app.route("/api/conversations", methods=["GET"])
def api_conversations():
    """获取当前用户的会话列表"""
    try:
        user_id = session.get("user_id")
        conversations = get_all_conversations(user_id=user_id)
        return jsonify({"success": True, "conversations": conversations})
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "conversations": []})


@app.route("/api/conversations", methods=["POST"])
def api_save_conversation():
    """保存会话元信息（标题等），并关联当前用户"""
    data = request.get_json() or {}
    session_id = data.get("session_id")
    title = data.get("title", "")
    if not session_id:
        return jsonify({"success": False, "message": "缺少 session_id"}), 400
    user_id = session.get("user_id")
    try:
        save_conversation_meta(session_id, title, user_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/conversations/<session_id>", methods=["DELETE"])
def api_delete_conversation(session_id):
    """删除会话，并清理相关 Redis 数据"""
    try:
        client = _get_redis_client()
        # 清理终止标记
        client.delete(f"conversation:{session_id}:stopped")
        client.delete(f"conversation:{session_id}:stopped_partial")
        # 递减话题偏好
        user_id = session.get("user_id")
        if user_id:
            memory_manager.decrement_query_topics(str(user_id), session_id)
        delete_conversation(session_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    kb_thread = threading.Thread(target=_start_kb_service, daemon=True)
    kb_thread.start()
    print(f"电商运营智能助手服务启动: http://{config.flask_host}:5001")
    print(f"知识库管理服务启动: http://{config.flask_host}:{config.flask_port}")
    app.run(host=config.flask_host, port=5001, debug=False)
