"""해동 코드 - 코딩 에이전트 도구 정의 및 실행."""
from __future__ import annotations

import os
import glob as glob_mod
import subprocess
import re


# --- 도구 스키마 (Ollama tool calling 형식) ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "파일 내용을 읽어 반환합니다. 줄 번호가 포함됩니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "읽을 파일 경로 (상대 또는 절대)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "파일을 생성하거나 덮어씁니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "파일 경로"},
                    "content": {"type": "string", "description": "파일에 쓸 내용"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "파일 내 특정 문자열을 다른 문자열로 치환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "파일 경로"},
                    "old_text": {"type": "string", "description": "찾을 문자열"},
                    "new_text": {"type": "string", "description": "교체할 문자열"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "glob 패턴으로 파일 목록을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob 패턴 (예: '**/*.py')"},
                    "path": {"type": "string", "description": "검색 시작 디렉토리 (기본: 현재 디렉토리)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "파일 내용에서 정규식 패턴을 검색합니다 (grep).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "검색할 정규식 패턴"},
                    "path": {"type": "string", "description": "검색 디렉토리 (기본: 현재 디렉토리)"},
                    "include": {"type": "string", "description": "파일 필터 glob (예: '*.py')"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "터미널 명령을 실행하고 결과를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "실행할 셸 명령"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "현재 Git 저장소의 상태를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Git diff를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "특정 파일 경로 (선택)"},
                    "staged": {"type": "boolean", "description": "staged 변경사항만 볼지 여부"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "변경사항을 스테이징하고 커밋합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "커밋 메시지"},
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "스테이징할 파일 목록 (비어있으면 전체)",
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "최근 Git 커밋 로그를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "표시할 커밋 수 (기본: 10)"},
                },
            },
        },
    },
]

# 사용자 확인이 필요한 위험 도구
DANGEROUS_TOOLS = {"write_file", "edit_file", "run_command", "git_commit"}

# 읽기 전용 안전 도구
SAFE_TOOLS = {"read_file", "list_files", "search_files", "git_status", "git_diff", "git_log"}


def _resolve_path(path: str, cwd: str) -> str:
    """상대 경로를 절대 경로로 변환."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cwd, path))


def _run_git(args: list[str], cwd: str) -> str:
    """Git 명령 실행 헬퍼."""
    result = subprocess.run(
        ["git"] + args,
        capture_output=True, text=True, cwd=cwd, timeout=30,
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        err = result.stderr.strip()
        return f"[오류] {err}" if err else f"[오류] git 명령 실패 (exit {result.returncode})"
    return output if output else "(변경사항 없음)"


def execute_tool(name: str, args: dict, cwd: str) -> str:
    """도구를 실행하고 결과 문자열을 반환."""
    try:
        if name == "read_file":
            return _tool_read_file(args, cwd)
        elif name == "write_file":
            return _tool_write_file(args, cwd)
        elif name == "edit_file":
            return _tool_edit_file(args, cwd)
        elif name == "list_files":
            return _tool_list_files(args, cwd)
        elif name == "search_files":
            return _tool_search_files(args, cwd)
        elif name == "run_command":
            return _tool_run_command(args, cwd)
        elif name == "git_status":
            return _run_git(["status", "--short"], cwd)
        elif name == "git_diff":
            return _tool_git_diff(args, cwd)
        elif name == "git_commit":
            return _tool_git_commit(args, cwd)
        elif name == "git_log":
            count = args.get("count", 10)
            return _run_git(["log", f"--oneline", f"-{count}"], cwd)
        else:
            return f"[오류] 알 수 없는 도구: {name}"
    except Exception as e:
        return f"[오류] {e}"


def _tool_read_file(args: dict, cwd: str) -> str:
    path = _resolve_path(args["path"], cwd)
    if not os.path.exists(path):
        return f"[오류] 파일이 존재하지 않습니다: {args['path']}"
    if os.path.isdir(path):
        return f"[오류] 디렉토리입니다. list_files를 사용하세요: {args['path']}"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
    if len(numbered) > 500:
        numbered = numbered[:500]
        numbered.append(f"... (+{len(lines) - 500}줄 더)")
    return "".join(numbered)


def _tool_write_file(args: dict, cwd: str) -> str:
    path = _resolve_path(args["path"], cwd)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(args["content"])
    return f"파일 작성 완료: {args['path']}"


def _tool_edit_file(args: dict, cwd: str) -> str:
    path = _resolve_path(args["path"], cwd)
    if not os.path.exists(path):
        return f"[오류] 파일이 존재하지 않습니다: {args['path']}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    old_text = args["old_text"]
    if old_text not in content:
        return f"[오류] 해당 텍스트를 찾을 수 없습니다."
    count = content.count(old_text)
    new_content = content.replace(old_text, args["new_text"], 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    msg = f"수정 완료: {args['path']}"
    if count > 1:
        msg += f" (첫 번째 일치만 교체, 총 {count}개 발견)"
    return msg


def _tool_list_files(args: dict, cwd: str) -> str:
    base = _resolve_path(args.get("path", "."), cwd)
    pattern = os.path.join(base, args["pattern"])
    files = sorted(glob_mod.glob(pattern, recursive=True))
    if not files:
        return "(일치하는 파일 없음)"
    rel = [os.path.relpath(f, cwd) for f in files]
    if len(rel) > 100:
        rel = rel[:100]
        rel.append(f"... (+{len(files) - 100}개 더)")
    return "\n".join(rel)


def _tool_search_files(args: dict, cwd: str) -> str:
    search_path = _resolve_path(args.get("path", "."), cwd)
    pattern = args["pattern"]
    include = args.get("include")

    cmd = ["grep", "-rn", "--color=never"]
    if include:
        cmd.extend(["--include", include])
    cmd.extend([pattern, search_path])

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, timeout=30,
    )
    output = result.stdout.strip()
    if not output:
        return "(일치하는 결과 없음)"
    lines = output.split("\n")
    if len(lines) > 50:
        lines = lines[:50]
        lines.append("... (결과가 더 있음)")
    # 절대 경로를 상대 경로로 변환
    result_lines = []
    for line in lines:
        result_lines.append(line.replace(cwd + "/", ""))
    return "\n".join(result_lines)


def _tool_run_command(args: dict, cwd: str) -> str:
    command = args["command"]
    result = subprocess.run(
        command, shell=True,
        capture_output=True, text=True, cwd=cwd, timeout=60,
    )
    output = ""
    if result.stdout.strip():
        output += result.stdout.strip()
    if result.stderr.strip():
        if output:
            output += "\n"
        output += f"[stderr] {result.stderr.strip()}"
    if result.returncode != 0:
        output += f"\n[exit code: {result.returncode}]"
    return output if output else "(출력 없음)"


def _tool_git_diff(args: dict, cwd: str) -> str:
    cmd = ["diff"]
    if args.get("staged"):
        cmd.append("--staged")
    path = args.get("path")
    if path:
        cmd.extend(["--", path])
    return _run_git(cmd, cwd)


def _tool_git_commit(args: dict, cwd: str) -> str:
    files = args.get("files", [])
    if files:
        add_result = _run_git(["add"] + files, cwd)
        if add_result.startswith("[오류]"):
            return add_result
    else:
        add_result = _run_git(["add", "-A"], cwd)
        if add_result.startswith("[오류]"):
            return add_result

    return _run_git(["commit", "-m", args["message"]], cwd)
