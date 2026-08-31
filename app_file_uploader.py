"""
知识库更新服务 - 入口已移至 knowledge_manage/kb_app.py
保留此文件仅为向后兼容，独立启动时自动委托给 knowledge_manage.kb_app
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
os.environ["GLOG_minloglevel"] = "2"

from knowledge_manage.kb_app import app
import config_data as config

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    print(f"知识库更新服务启动: http://{config.flask_host}:{config.flask_port}")
    app.run(host=config.flask_host, port=config.flask_port, debug=False)
