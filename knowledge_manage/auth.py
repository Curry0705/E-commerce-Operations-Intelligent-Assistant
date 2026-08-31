"""知识库管理员认证"""
import bcrypt
from flask import Blueprint, request, jsonify, session
from knowledge_manage.db import query_admin_by_username, query_admin_by_id, insert_admin, update_admin_password

kb_auth_bp = Blueprint("kb_auth", __name__)


@kb_auth_bp.route("/api/kb/login", methods=["POST"])
def api_login():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请提供登录信息"}), 400

    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"success": False, "message": "用户名和密码不能为空"}), 400

    admin = query_admin_by_username(username)
    if admin is None:
        return jsonify({"success": False, "message": "用户名或密码错误"}), 401

    stored_hash = admin["password"]
    if not bcrypt.checkpw(password.encode(), stored_hash.encode()):
        return jsonify({"success": False, "message": "用户名或密码错误"}), 401

    session.permanent = False
    session["admin_id"] = admin["admin_id"]
    session["admin_username"] = admin["username"]

    return jsonify({
        "success": True,
        "message": "登录成功",
        "admin": {
            "admin_id": admin["admin_id"],
            "username": admin["username"],
            "avatar": admin.get("avatar", ""),
        },
    })


@kb_auth_bp.route("/api/kb/register", methods=["POST"])
def api_register():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请提供注册信息"}), 400

    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    confirm_password = (data.get("confirm_password") or "").strip()
    phone = (data.get("phone") or "").strip()
    email = (data.get("email") or "").strip()

    if not username or not password:
        return jsonify({"success": False, "message": "用户名和密码不能为空"}), 400

    if password != confirm_password:
        return jsonify({"success": False, "message": "两次密码输入不一致"}), 400

    if len(password) < 6:
        return jsonify({"success": False, "message": "密码长度不能少于 6 位"}), 400

    if query_admin_by_username(username) is not None:
        return jsonify({"success": False, "message": "该用户名已被注册"}), 409

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    admin_id = insert_admin(username, hashed, phone, email)

    return jsonify({
        "success": True,
        "message": "注册成功",
        "admin_id": admin_id,
    })


@kb_auth_bp.route("/api/kb/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True, "message": "已退出登录"})


@kb_auth_bp.route("/api/kb/status", methods=["GET"])
def api_status():
    if "admin_id" in session:
        admin = query_admin_by_id(session["admin_id"])
        return jsonify({
            "success": True,
            "logged_in": True,
            "admin": {
                "admin_id": session["admin_id"],
                "username": session["admin_username"],
                "avatar": admin.get("avatar", "") if admin else "",
            },
        })
    return jsonify({"success": True, "logged_in": False})


@kb_auth_bp.route("/api/kb/alter-password", methods=["POST"])
def api_alter_password():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请提供密码信息"}), 400

    username = (data.get("username") or "").strip()
    new_password = (data.get("new_password") or "").strip()
    confirm_password = (data.get("confirm_password") or "").strip()

    if not username or not new_password:
        return jsonify({"success": False, "message": "请填写用户名和新密码"}), 400

    if new_password != confirm_password:
        return jsonify({"success": False, "message": "两次新密码输入不一致"}), 400

    if len(new_password) < 6:
        return jsonify({"success": False, "message": "新密码长度不能少于 6 位"}), 400

    admin = query_admin_by_username(username)
    if admin is None:
        return jsonify({"success": False, "message": "用户名不存在"}), 404

    new_hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    update_admin_password(username, new_hashed)

    return jsonify({"success": True, "message": "密码重置成功"})
