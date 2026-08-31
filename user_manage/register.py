"""用户注册"""
import bcrypt
from flask import Blueprint, request, jsonify
from user_manage.db import query_user_by_username, insert_user

register_bp = Blueprint("register", __name__)


@register_bp.route("/api/auth/register", methods=["POST"])
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

    if query_user_by_username(username) is not None:
        return jsonify({"success": False, "message": "该用户名已被注册"}), 409

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_id = insert_user(username, hashed, phone, email)

    return jsonify({
        "success": True,
        "message": "注册成功",
        "user_id": user_id,
    })
