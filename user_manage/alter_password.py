"""忘记密码 / 重置密码"""
import bcrypt
from flask import Blueprint, request, jsonify
from user_manage.db import query_user_by_username, update_password

alter_password_bp = Blueprint("alter_password", __name__)


@alter_password_bp.route("/api/auth/alter-password", methods=["POST"])
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

    user = query_user_by_username(username)
    if user is None:
        return jsonify({"success": False, "message": "用户名不存在"}), 404

    new_hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    update_password(username, new_hashed)

    return jsonify({"success": True, "message": "密码重置成功"})
