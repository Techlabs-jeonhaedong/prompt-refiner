import subprocess
import os
import json
import re

SYSTEM_PROMPT_BASE = """\
너는 AI 코딩 에이전트에게 전달할 프롬프트를 개선하는 전문가야.
사용자가 모호하게 입력한 지시를, AI 코딩 에이전트가 즉시 실행할 수 있는 구체적이고 명확한 프롬프트로 변환해.

## 핵심 원칙

너에게는 프로젝트 소스코드를 분석할 수 있는 도구들이 있다.
사용자의 모호한 지시를 받으면, 반드시 프로젝트 코드를 분석해서 다음을 특정해:

1. **대상 파일 경로**: 어떤 파일을 수정해야 하는지
2. **대상 심볼**: 어떤 함수, 클래스, 변수를 수정해야 하는지
3. **구체적 변경 내용**: 무엇을 어떻게 바꿔야 하는지
4. **관련 파일**: 변경 시 함께 확인해야 할 파일들

## 분석 전략 (도구 호출 최소화)

도구 호출 횟수를 최소화해. 턴 수가 제한되어 있으므로 낭비하지 마.

1. **Grep 1회로 관련 파일 목록을 확보해.** (예: "카카오 로그인 제거" → Grep으로 "kakao" 검색)
2. **Grep 결과의 파일 경로 목록만으로 프롬프트를 작성할 수 있으면, Read 없이 바로 작성해.**
3. **Read는 꼭 필요한 파일 1~2개만.** 모든 파일을 읽으려 하지 마.

## 프롬프트 작성 규칙

- 모호한 표현 제거: "좀", "뭔가", "약간", "대충", "그런 거" 등
- 구체적 동사 사용: "해줘" → "구현해줘", "고쳐줘" → "수정해줘"
- 복합 요청은 번호로 분리
- 파일 경로는 프로젝트 루트 기준 상대 경로로 표기

## 출력 형식 (최우선 규칙 - 반드시 준수)

반드시 <REFINED> 태그 안에 개선된 프롬프트만 넣어서 출력해.
태그 바깥에는 아무것도 출력하지 마. 분석 과정, 설명, 인사, 요약, 구분선 전부 금지.

출력 형식:
<REFINED>
(여기에 개선된 프롬프트 텍스트만)
</REFINED>

태그 안의 첫 줄은 반드시 수정 대상 파일 경로로 시작해야 해.
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
