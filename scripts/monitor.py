#!/usr/bin/env python
"""RAG Project Monitor --- 扫描项目代码变更、评估结果、日志异常并生成摘要。

纯标准库实现，不依赖 langchain/ragas 等重型库，可在 venv 外直接运行。
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_FILE = PROJECT_ROOT / ".monitor_snapshot.json"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def run_git(*args: str) -> str | None:
    """执行 git 命令并返回 stdout；失败返回 None。"""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def load_snapshot() -> dict:
    """加载上次快照。"""
    if SNAPSHOT_FILE.exists():
        try:
            return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_snapshot(data: dict) -> None:
    """保存快照到磁盘。"""
    SNAPSHOT_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 检测模块
# ---------------------------------------------------------------------------

def check_git() -> bool:
    """Git 状态：分支、未提交变更、最近 commit。返回是否有需要关注的变更。"""
    print("## Git 状态")

    branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    print(f"- 当前分支: {branch or 'N/A'}")

    if branch is None:
        print("- [WARN] 不是 Git 仓库或 Git 不可用")
        print()
        return False

    status = run_git("status", "--short")
    if status:
        lines = status.split("\n")
        modified = [l for l in lines if l.strip() and not l.startswith("??")]
        untracked = [l for l in lines if l.startswith("??")]
        print(f"- 未提交修改: {len(modified)} 个文件")
        for l in modified[:5]:
            print(f"    {l.strip()}")
        if len(modified) > 5:
            print(f"    ... 及其他 {len(modified) - 5} 个文件")
        print(f"- 未跟踪文件: {len(untracked)} 个文件")
        for l in untracked[:5]:
            print(f"    {l.strip()}")
        if len(untracked) > 5:
            print(f"    ... 及其他 {len(untracked) - 5} 个文件")
    else:
        print("- 工作区干净 [OK]")

    log = run_git("log", "--oneline", "-3")
    if log:
        print("- 最近提交:")
        for line in log.split("\n"):
            print(f"    {line}")

    has_uncommitted = bool(status and status.strip())
    print()
    return has_uncommitted


def check_py_files(snapshot: dict) -> tuple[bool, dict]:
    """对比 .py 文件时间戳，检测增/删/改。返回 (有变更, 更新后的 py 快照)。"""
    print("## Python 文件变更")

    prev_py = snapshot.get("py_files", {})
    current_py = {}
    changed: list[str] = []
    new_files: list[str] = []
    deleted: list[str] = []

    for py_file in sorted(PROJECT_ROOT.glob("*.py")):
        mtime = py_file.stat().st_mtime
        current_py[py_file.name] = mtime

        if py_file.name in prev_py:
            if abs(mtime - prev_py[py_file.name]) > 0.001:
                changed.append(py_file.name)
        else:
            new_files.append(py_file.name)

    for name in sorted(prev_py):
        if name not in current_py:
            deleted.append(name)

    if changed:
        print(f"- 修改: {len(changed)} 个文件")
        for f in changed:
            print(f"    {f}")
    if new_files:
        print(f"- 新增: {len(new_files)} 个文件")
        for f in new_files:
            print(f"    {f}")
    if deleted:
        print(f"- 删除: {len(deleted)} 个文件")
        for f in deleted:
            print(f"    {f}")
    if not (changed or new_files or deleted):
        print("- 无变更 [OK]")

    has_changes = bool(changed or new_files or deleted)
    print()
    return has_changes, current_py


def check_reports(snapshot: dict) -> tuple[bool, list[str]]:
    """扫描 reports/ 目录的 RAGAS JSON 报告。返回 (有问题, 本次报告列表)。"""
    print("## 评估报告")

    reports_dir = PROJECT_ROOT / "reports"
    if not reports_dir.is_dir():
        print("- reports/ 目录不存在")
        print()
        return False, []

    reports = sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        print("- 无报告文件")
        print()
        return False, []

    prev_seen = set(snapshot.get("reports_seen", []))
    print(f"- 共 {len(reports)} 份报告")

    has_issues = False
    for report in reports[:8]:
        name = report.name
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"    {name} -- 解析失败: {e}")
            has_issues = True
            continue

        new_flag = " [NEW]" if name not in prev_seen else ""
        print(f"    {name}{new_flag}")

        # 遍历顶层 section，提取 scores
        for section_key, section in data.items():
            if not isinstance(section, dict):
                continue
            scores = section.get("scores")
            if not isinstance(scores, dict):
                continue
            label = section.get("label", section_key)
            valid = {k: v for k, v in scores.items() if v is not None}
            nulls = [k for k, v in scores.items() if v is None]
            if valid:
                metric_str = "  ".join(f"{k}: {v:.4f}" for k, v in valid.items())
                print(f"      [{label}] {metric_str}")
            if nulls:
                print(f"      [{label}] [WARN] 空指标: {', '.join(nulls)}")
                has_issues = True
            # elapsed 可能在 section 级别
            elapsed = section.get("elapsed_s")
            if elapsed is not None:
                print(f"      耗时: {elapsed:.1f}s")

    if len(reports) > 8:
        print(f"    ... 及其他 {len(reports) - 8} 份报告")

    report_names = [r.name for r in reports]
    print()
    return has_issues, report_names


def check_logs() -> bool:
    """解析 .log 文件中的 error/traceback/warning 行。返回是否有异常。"""
    print("## 日志文件")

    log_files = sorted((PROJECT_ROOT / "logs").glob("*.log"))
    if not log_files:
        print("- 无日志文件")
        print()
        return False

    has_issues = False
    for log_file in log_files:
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"- {log_file.name}: 无法读取 ({e})")
            continue

        lines = text.splitlines()
        total_lines = len(lines)
        error_lines = [l.strip() for l in lines if _is_error(l)]
        warn_lines = [l.strip() for l in lines if "warn" in l.lower()]

        print(f"- {log_file.name}: {total_lines} 行, "
              f"error/traceback {len(error_lines)}, warning {len(warn_lines)}")

        if error_lines:
            has_issues = True
            print("   error/traceback 示例:")
            for e_line in error_lines[:3]:
                display = e_line[:150] + ("..." if len(e_line) > 150 else "")
                print(f"     {display}")

        if warn_lines:
            print("   warning 示例:")
            for w_line in warn_lines[:2]:
                display = w_line[:150] + ("..." if len(w_line) > 150 else "")
                print(f"     {display}")

    print()
    return has_issues


def _is_error(line: str) -> bool:
    """判断一行是否为错误相关行。"""
    lower = line.lower()
    return "error" in lower or "traceback" in lower or "exception" in lower


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print(f"RAG 项目监控报告 -- {now}")
    print("=" * 60)
    print()

    snapshot = load_snapshot()

    git_issue = check_git()
    py_changed, py_snapshot = check_py_files(snapshot)
    report_issue, report_list = check_reports(snapshot)
    log_issue = check_logs()

    # 更新并保存快照
    snapshot["py_files"] = py_snapshot
    snapshot["reports_seen"] = report_list
    snapshot["last_check"] = now
    save_snapshot(snapshot)

    # 总状态
    issues_count = sum([git_issue, py_changed, report_issue, log_issue])
    print("=" * 60)
    if issues_count == 0:
        print("[OK] 状态: 一切正常")
    elif issues_count <= 2:
        print("[WARN] 状态: 需要关注")
    else:
        print("[ALERT] 状态: 有异常，建议检查")
    print("=" * 60)

    return 0 if issues_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
