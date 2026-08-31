"""
知识库更新服务（带认证）
基于 Flask + Vue 3 完成知识库更新服务

由 app_qa.py 在后台线程启动，或独立启动：python -m knowledge_manage.kb_app
访问地址: http://127.0.0.1:5000
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
os.environ["GLOG_minloglevel"] = "2"

from flask import Flask, request, jsonify, send_from_directory, session, redirect
from knowledge_base import KnowledgeBaseService
from knowledge_manage.auth import kb_auth_bp
from knowledge_manage.admin_center import admin_bp
import config_data as config

# 静态文件目录：相对 kb_app.py 的上级目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATIC_ADMINS = os.path.join(_PROJECT_ROOT, "static", "admins")

app = Flask(__name__)
app.secret_key = config.flask_secret_key
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

app.register_blueprint(kb_auth_bp)
app.register_blueprint(admin_bp)

kb_service = KnowledgeBaseService()

ALLOWED_EXTENSIONS = {"txt", "pdf", "xlsx", "docx", "pptx", "md"}

# 公开路径
PUBLIC_PATHS = {
    "/", "/api/kb/login", "/api/kb/register", "/api/kb/alter-password", "/api/kb/status",
    "/kb_login.html", "/kb_register.html", "/kb_alter_password.html",
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.before_request
def check_auth():
    if request.path in PUBLIC_PATHS or request.path.startswith("/static/"):
        return None
    if "admin_id" not in session:
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "message": "请先登录", "redirect": "/kb_login.html"}), 401
        return redirect("/kb_login.html")


@app.route("/")
def index():
    if "admin_id" not in session:
        return send_from_directory(_STATIC_ADMINS, "kb_login.html")
    return send_from_directory(_STATIC_ADMINS, "file_uploader.html")


@app.route("/kb_login.html")
def login_page():
    return send_from_directory(_STATIC_ADMINS, "kb_login.html")


@app.route("/kb_alter_password.html")
def alter_password_page():
    return send_from_directory(_STATIC_ADMINS, "kb_alter_password.html")


@app.route("/kb_register.html")
def register_page():
    return send_from_directory(_STATIC_ADMINS, "kb_register.html")


@app.route("/admin_center.html")
def admin_center_page():
    if "admin_id" not in session:
        return redirect("/kb_login.html")
    return send_from_directory(_STATIC_ADMINS, "admin_center.html")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "未选择文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "文件名为空"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": f"不支持的文件格式，仅支持: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    try:
        file_bytes = file.read()
        result = kb_service.upload_by_bytes(file_bytes, file.filename)
        return jsonify({"success": True, "message": result, "filename": file.filename})
    except Exception as e:
        return jsonify({"success": False, "message": f"上传失败: {str(e)}"}), 500


@app.route("/api/upload_batch", methods=["POST"])
def api_upload_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"success": False, "message": "未选择文件"}), 400

    results = []
    for f in files:
        if f.filename == "":
            continue
        if not allowed_file(f.filename):
            results.append({
                "filename": f.filename,
                "success": False,
                "message": f"不支持的文件格式，仅支持: {', '.join(ALLOWED_EXTENSIONS)}",
            })
            continue
        try:
            file_bytes = f.read()
            msg = kb_service.upload_by_bytes(file_bytes, f.filename)
            results.append({"filename": f.filename, "success": True, "message": msg})
        except Exception as e:
            results.append({"filename": f.filename, "success": False, "message": f"上传失败: {str(e)}"})

    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count

    return jsonify({
        "success": fail_count == 0,
        "message": f"批量上传完成: 成功 {success_count} 个, 失败 {fail_count} 个",
        "total": len(results),
        "success_count": success_count,
        "fail_count": fail_count,
        "results": results,
    })


@app.route("/api/documents", methods=["GET"])
def api_list_documents():
    try:
        filenames = kb_service.get_all_filenames()
        return jsonify({"success": True, "documents": filenames})
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "documents": []})


@app.route("/api/documents/<path:filename>", methods=["DELETE"])
def api_delete_document(filename):
    try:
        result = kb_service.delete_by_filename(filename)
        return jsonify({"success": True, "message": result})
    except Exception as e:
        return jsonify({"success": False, "message": f"删除失败: {str(e)}"}), 500


@app.route("/api/images/reset", methods=["POST"])
def api_reset_image_collection():
    try:
        result = kb_service.reset_image_collection()
        return jsonify({"success": True, "message": result})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/documents/clear_all", methods=["DELETE"])
def api_clear_all():
    try:
        result = kb_service.clear_all()
        return jsonify({"success": True, "message": result})
    except Exception as e:
        return jsonify({"success": False, "message": f"清空失败: {str(e)}"}), 500


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    print(f"知识库更新服务启动: http://{config.flask_host}:{config.flask_port}")
    app.run(host=config.flask_host, port=config.flask_port, debug=False)
