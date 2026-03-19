import subprocess
import os
import json
import re

SYSTEM_PROMPT_BASE = """\
You are an expert at refining prompts for AI coding agents.
Transform vague user instructions into specific, actionable prompts that an AI coding agent can execute immediately.

## Core Principles

You have tools to analyze the project source code.
When you receive a vague instruction, you MUST analyze the project code to identify:

1. **Target file paths**: Which files need to be modified
2. **Target symbols**: Which functions, classes, or variables need to be modified
3. **Related files**: Which files should be checked alongside the changes

## Analysis Strategy (Minimize Tool Calls)

Minimize the number of tool calls. Turns are limited — do not waste them.

1. **Use a single Grep call to find relevant files.** (e.g., "remove Kakao login" → Grep for "kakao")
2. **If the file path list from Grep is sufficient to write the prompt, skip Read entirely.**
3. **Only Read 1-2 files when absolutely necessary.** Do not attempt to read every file.

## Prompt Writing Rules

- Eliminate vague expressions: "kind of", "something like", "sort of", "roughly", "that thing", etc.
- Use specific verbs: "do it" → "implement", "fix it" → "modify the return value of"
- Split compound requests into numbered steps
- Use relative paths from the project root for all file paths
- Transform every ambiguous instruction into a precise, unambiguous sentence. \
Each sentence must clearly state WHAT to do, WHERE to do it, and the expected RESULT. \
Example: "make the button nicer" → "Change the background color of the submit button in src/components/LoginForm.tsx to #3B82F6 (blue-500) and increase the border-radius to 8px"

## Strict Prohibitions

- **NEVER include source code in the output.** Do not show code snippets, diffs, before/after code examples, or implementation details. \
The refined prompt must describe WHAT to change in natural language, not HOW to write the code.
- **NEVER use code blocks (``` ```) in the output.**

## Output Format (Highest Priority Rule — MUST Follow)

You MUST output only the refined prompt inside <REFINED> tags.
Output NOTHING outside the tags. No analysis, no explanations, no greetings, no summaries, no separators.

The output MUST follow this structure:
1. **File list**: Start with "## 수정 대상 파일" followed by a bullet list of all files that need modification
2. **Instructions**: Then list the specific changes as numbered steps in natural language

Output format:
<REFINED>
## 수정 대상 파일
- path/to/file1.py
- path/to/file2.ts

1. (specific instruction in natural language — no code)
2. (specific instruction in natural language — no code)
</REFINED>
"""

BASE_TOOLS = ["Read", "Glob", "Grep"]

MAX_TURNS = 15


def _claude_home() -> str:
    """Claude 설정 디렉토리 경로 반환."""
    return os.path.expanduser("~/.claude")


def _has_serena_in_json(path: str, key: str = "mcpServers") -> bool:
    """JSON 파일에서 serena 관련 항목이 있는지 확인."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        entries = data.get(key, {})
        return any("serena" in name.lower() for name in entries)
    except (json.JSONDecodeError, IOError):
        return False


def detect_serena(cwd: str) -> bool:
    """프로젝트 또는 글로벌에 Serena가 설치되어 있는지 확인."""
    # 1. 프로젝트 로컬 MCP 설정
    for config_name in [".mcp.json", ".mcp/config.json"]:
        if _has_serena_in_json(os.path.join(cwd, config_name)):
            return True

    claude_home = _claude_home()

    # 2. 글로벌 MCP 설정
    if _has_serena_in_json(os.path.join(claude_home, ".mcp.json")):
        return True

    # 3. 글로벌 플러그인
    if _has_serena_in_json(
        os.path.join(claude_home, "plugins", "installed_plugins.json"),
        key="plugins",
    ):
        return True

    return False


def scan_project_tree(root: str, max_files: int = 200) -> str:
    """프로젝트 디렉토리 트리를 문자열로 반환."""
    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".dart_tool",
        "build", "dist", ".next", ".nuxt", "venv", ".venv",
        "env", ".env", ".idea", ".vscode", "Pods", ".gradle",
        ".build", "DerivedData", "coverage", ".pytest_cache",
    }
    skip_exts = {
        ".pyc", ".pyo", ".class", ".o", ".so", ".dylib",
        ".lock", ".png", ".jpg", ".jpeg", ".gif", ".ico",
        ".woff", ".woff2", ".ttf", ".eot", ".svg",
        ".mp3", ".mp4", ".mov", ".avi",
    }

    lines = []
    count = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        dirnames.sort()

        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1

        if depth <= 3:
            prefix = "  " * depth
            dirname = os.path.basename(dirpath) + "/" if rel != "." else ""
            if dirname:
                lines.append(f"{prefix}{dirname}")

        for f in sorted(filenames):
            ext = os.path.splitext(f)[1].lower()
            if ext in skip_exts:
                continue
            count += 1
            if count > max_files:
                lines.append(f"  ... (+{count - max_files} more files)")
                return "\n".join(lines)
            if depth <= 3:
                prefix = "  " * (depth + 1)
                lines.append(f"{prefix}{f}")

    return "\n".join(lines)


def refine(prompt: str) -> str:
    cwd = os.getcwd()
    tree = scan_project_tree(cwd)

    context_prompt = f"""\
## 현재 프로젝트 정보

작업 디렉토리: {cwd}

### 프로젝트 구조
```
{tree}
```

## 사용자의 원본 지시

{prompt}
"""

    result = subprocess.run(
        [
            "claude",
            "-p",
            "--system-prompt", SYSTEM_PROMPT_BASE,
            "--no-session-persistence",
            "--allowedTools", *BASE_TOOLS,
            "--max-turns", str(MAX_TURNS),
            "--model", "sonnet",
            context_prompt,
        ],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude Code 실행 실패: {result.stderr.strip()}")

    return _strip_meta(result.stdout.strip())


def _strip_meta(text: str) -> str:
    """<REFINED> 태그 안의 내용만 추출. 태그가 없으면 기존 폴백 로직 사용."""
    # 1차: <REFINED> 태그 추출
    match = re.search(r"<REFINED>\s*\n?(.*?)\s*</REFINED>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 폴백: 태그 없이 출력된 경우 앞쪽 메타 라인 제거
    lines = text.split("\n")

    meta_patterns = re.compile(
        r"^("
        r"-{2,}|"                           # --- 구분선
        r"#{1,}\s|"                          # ## 마크다운 헤더
        r".*(?:분석|파악|확인|완료|출력|생성|개선된|프롬프트를|결과)"
        r")",
        re.IGNORECASE,
    )

    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if meta_patterns.match(stripped):
            start = i + 1
        else:
            break

    cleaned = "\n".join(lines[start:]).strip()
    return cleaned if cleaned else text


def execute(refined_prompt: str) -> int:
    """개선된 프롬프트를 Opus 모델로 대화형 실행. 터미널을 그대로 넘겨서 사용자가 직접 상호작용."""
    cwd = os.getcwd()
    proc = subprocess.run(
        [
            "claude",
            "--dangerously-skip-permissions",
            "--model", "opus",
            refined_prompt,
        ],
        cwd=cwd,
    )
    return proc.returncode
