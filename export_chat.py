"""
导出聊天记录：将 JSONL 格式的会话日志转换为可读的文本文件
"""
import json
import os

jsonl_path = r"C:\Users\20678\.claude\projects\D--Pythoncode-RAG\8b1f983b-b6e5-4890-9947-c1e9e8ab249f.jsonl"
output_path = r"D:\Pythoncode\RAG\chat_history.txt"

lines = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            msg_type = data.get("type", "")

            if msg_type == "user" and "message" in data:
                content = data["message"].get("content", [])
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        lines.append(f"\n{'='*60}")
                        lines.append(f"[用户] {data.get('timestamp', '')}")
                        lines.append(f"{'='*60}")
                        lines.append(item["text"])

            elif msg_type == "assistant" and "message" in data:
                content = data["message"].get("content", [])
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        lines.append(f"\n{'='*60}")
                        lines.append(f"[AI] {data.get('timestamp', '')}")
                        lines.append(f"{'='*60}")
                        lines.append(item["text"])
        except:
            pass

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"导出完成：{len(lines)} 行 → {output_path}")
