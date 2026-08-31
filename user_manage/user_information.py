"""用户个人信息：查看 + 修改"""
import json
from flask import Blueprint, request, jsonify, session
from user_manage.db import query_user_by_id, query_user_by_username, update_user_info

user_information_bp = Blueprint("user_information", __name__)


@user_information_bp.route("/api/auth/user-info", methods=["GET", "PUT"])
def api_user_info():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "请先登录"}), 401

    user_id = session["user_id"]

    if request.method == "GET":
        user = query_user_by_id(user_id)
        if user is None:
            return jsonify({"success": False, "message": "用户不存在"}), 404
        return jsonify({
            "success": True,
            "user": {
                "user_id": user["user_id"],
                "username": user["username"],
                "phone": user.get("phone", ""),
                "email": user.get("email", ""),
                "avatar": user.get("avatar", ""),
                "tags": user.get("tags", ""),
            },
        })

    # PUT：更新个人信息
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请提供更新内容"}), 400

    fields = {}
    for key in ("phone", "email", "avatar", "tags"):
        if key in data:
            fields[key] = data[key]

    if not fields:
        return jsonify({"success": False, "message": "没有可更新的字段"}), 400

    username = session.get("username", "")
    ok = update_user_info(username, fields)
    if ok:
        return jsonify({"success": True, "message": "个人信息已更新"})
    return jsonify({"success": False, "message": "更新失败"}), 500


@user_information_bp.route("/api/auth/user-avatar", methods=["GET"])
def api_user_avatar():
    """获取当前用户的头像和用户名（用于侧边栏显示）"""
    if "user_id" not in session:
        return jsonify({"success": False, "message": "请先登录"}), 401

    user = query_user_by_id(session["user_id"])
    if user is None:
        return jsonify({"success": False, "message": "用户不存在"}), 404
    return jsonify({
        "success": True,
        "user": {
            "username": user["username"],
            "avatar": user.get("avatar", ""),
        },
    })
