"""
=========== 对话学习记录 → 思源笔记 ===========

用法：
    python save_to_siyuan.py             # 交互式输入
    python save_to_siyuan.py "今天学了RAG的向量化原理"  # 命令行直接传

会自动在思源笔记中创建一条日记，标题格式：2026-06-05 AI 学习记录
"""

import sys
import requests
from datetime import datetime

# ===== 思源 API 配置 =====
SIYUAN_API = "http://127.0.0.1:6806"
SIYUAN_TOKEN = "pcmnpn14zzwj1l9m"    # 在思源 → 设置 → 关于 → API Token 中查看
NOTEBOOK_ID = ""                     # 留空则自动使用第一个笔记本


def get_default_notebook():
    """获取第一个笔记本的 ID"""
    resp = requests.post(
        f"{SIYUAN_API}/api/notebook/lsNotebooks",
        headers={"Authorization": f"Token {SIYUAN_TOKEN}"},
    )
    notebooks = resp.json()["data"]["notebooks"]
    return notebooks[0]["id"]


def create_daily_doc(content: str):
    """在思源中创建一条日记"""
    today = datetime.now().strftime("%Y-%m-%d")
    title = f"{today} AI 学习记录"

    # 创建文档
    resp = requests.post(
        f"{SIYUAN_API}/api/filetree/createDailyNote",
        headers={"Authorization": f"Token {SIYUAN_TOKEN}"},
        json={"notebook": NOTEBOOK_ID or get_default_notebook()},
    )
    note_id = resp.json()["data"]["id"]

    # 写入内容
    resp = requests.put(
        f"{SIYUAN_API}/api/block/updateBlock",
        headers={"Authorization": f"Token {SIYUAN_TOKEN}"},
        json={
            "id": note_id,
            "dataType": "markdown",
            "data": f"# {title}\n\n{content}",
        },
    )
    if resp.status_code == 200:
        print(f"[完成] 已保存到思源：{title}")
    else:
        print(f"[失败] {resp.text}")


def save_to_md(content: str):
    """备用方案：保存为 .md 文件，手动拖入思源"""
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{today}-AI学习记录.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {today} AI 学习记录\n\n{content}")
    print(f"[完成] 已保存为 {filename}，可拖入思源笔记")


if __name__ == "__main__":
    # 获取内容
    args = sys.argv[1:]
    if args:
        content = " ".join(args)
    else:
        content = input("请输入学习总结内容：\n").strip()

    if not content:
        print("[错误] 内容为空")
        sys.exit(1)

    # 尝试写入思源，失败则保存为 md 文件
    try:
        create_daily_doc(content)
    except Exception as e:
        print(f"[提示] 思源 API 连接失败 ({e})，改用 md 文件")
        save_to_md(content)
