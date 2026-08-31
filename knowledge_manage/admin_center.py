"""管理员中心 API"""
from flask import Blueprint, request, jsonify, session
from knowledge_manage.db import (
    query_admin_by_id, update_admin_info,
    get_upload_records, get_document_stats, get_admin_count
)

admin_bp = Blueprint("admin_center", __name__)


@admin_bp.route("/api/admin/profile", methods=["GET"])
def get_profile():
    if "admin_id" not in session:
        return jsonify({"success": False, "message": "未登录"}), 401
    admin = query_admin_by_id(session["admin_id"])
    if not admin:
        return jsonify({"success": False, "message": "管理员不存在"}), 404
    return jsonify({
        "success": True,
        "admin": {
            "admin_id": admin["admin_id"],
            "username": admin["username"],
            "phone": admin.get("phone", ""),
            "email": admin.get("email", ""),
            "avatar": admin.get("avatar", ""),
            "tags": admin.get("tags", ""),
            "created_at": str(admin.get("created_at", "")),
        },
    })


@admin_bp.route("/api/admin/profile", methods=["PUT"])
def update_profile():
    if "admin_id" not in session:
        return jsonify({"success": False, "message": "未登录"}), 401
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "请提供更新信息"}), 400
    ok = update_admin_info(session["admin_username"], data)
    return jsonify({"success": ok, "message": "更新成功" if ok else "无有效更新字段"})


@admin_bp.route("/api/admin/stats", methods=["GET"])
def get_stats():
    if "admin_id" not in session:
        return jsonify({"success": False, "message": "未登录"}), 401
    stats = get_document_stats()
    stats["admin_count"] = get_admin_count()
    return jsonify({"success": True, "stats": stats})


@admin_bp.route("/api/admin/upload-records", methods=["GET"])
def get_records():
    if "admin_id" not in session:
        return jsonify({"success": False, "message": "未登录"}), 401
    records = get_upload_records()
    for r in records:
        if r.get("upload_time"):
            r["upload_time"] = str(r["upload_time"])
    return jsonify({"success": True, "records": records})
