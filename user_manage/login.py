"""登录 / 登出"""
import bcrypt
from flask import Blueprint, request, jsonify, session
from user_manage.db import query_user_by_username

login_bp = Blueprint("login", __name__)


@login_bp.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请提供登录信息"}), 400

    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"success": False, "message": "用户名和密码不能为空"}), 400

    user = query_user_by_username(username)
    if user is None:
        return jsonify({"success": False, "message": "用户名或密码错误"}), 401

    stored_hash = user["password"]
    if not bcrypt.checkpw(password.encode(), stored_hash.encode()):
        return jsonify({"success": False, "message": "用户名或密码错误"}), 401

    session["user_id"] = user["user_id"]
    session["username"] = user["username"]

    return jsonify({
        "success": True,
        "message": "登录成功",
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "avatar": user.get("avatar", ""),
        },
    })


@login_bp.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True, "message": "已退出登录"})


@login_bp.route("/api/auth/status", methods=["GET"])
def api_status():
    """返回当前登录状态"""
    if "user_id" in session:
        return jsonify({
            "success": True,
            "logged_in": True,
            "user": {
                "user_id": session["user_id"],
                "username": session["username"],
            },
        })
    return jsonify({"success": True, "logged_in": False})
