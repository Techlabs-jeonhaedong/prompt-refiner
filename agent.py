"""해동 코드 - 로컬 LLM 기반 코딩 에이전트 코어.

기본: 내장 모델 (mlx-lm, Apple Silicon 최적화)
옵션: 외부 API 서버 (LM Studio / Ollama) — --url 플래그로 활성화
"""
from __future__ import annotations

import json
import os
import re

from tools import TOOL_SCHEMAS, DANGEROUS_TOOLS, execute_tool

# --- 기본 설정 ---
DEFAULT_MODEL = "mlx-community/Qwen3-30B-A3B-4bit"

# --- 프롬프트 기반 도구 호출용 시스템 프롬프트 ---

_TOOL_DESCRIPTIONS = """
사용 가능한 도구 목록:

1. read_file(path): 파일 내용을 줄 번호와 함께 읽습니다.
2. write_file(path, content): 파일을 생성하거나 덮어씁니다.
3. edit_file(path, old_text, new_text): 파일 내 특정 문자열을 교체합니다.
4. list_files(pattern, path?): glob 패턴으로 파일 목록을 반환합니다. (예: "**/*.py")
5. search_files(pattern, path?, include?): 파일 내용에서 정규식 검색합니다 (grep).
6. run_command(command): 터미널 명령을 실행합니다.
7. git_status(): Git 상태를 확인합니다.
8. git_diff(path?, staged?): Git diff를 봅니다.
9. git_commit(message, files?): 변경사항을 커밋합니다.
10. git_log(count?): 최근 커밋 로그를 봅니다.
"""

SYSTEM_PROMPT = """\
You are Haedong Code, a powerful local AI coding agent.
You help users with software engineering tasks by reading, writing, and modifying code,
running commands, and managing git repositories.

## Rules
- Always respond in Korean (한국어).
- Be concise and direct.
- When modifying files, read them first to understand the context.
- Explain what you're doing before using tools.
- For destructive operations, clearly state what will change.
- Use relative paths from the project root.

## Tool Usage
{tool_descriptions}

도구를 사용하려면 다음 형식을 사용하세요:
<tool_call>
{{"name": "도구이름", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

한 번에 하나의 도구만 호출하세요.
도구 실행 결과는 <tool_result> 태그로 전달됩니다.
도구를 사용한 뒤에는 반드시 결과를 확인하고 사용자에게 설명하세요.

## Current Working Directory
{cwd}
"""


def _build_system_prompt(cwd: str) -> str:
    return SYSTEM_PROMPT.format(tool_descriptions=_TOOL_DESCRIPTIONS, cwd=cwd)


def _parse_tool_calls(text: str) -> list[dict]:
    """응답 텍스트에서 <tool_call> 태그를 파싱."""
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    results = []
    for match in matches:
        try:
            call = json.loads(match)
            if "name" in call:
                results.append(call)
        except json.JSONDecodeError:
            continue
    return results


class Agent:
    """로컬 LLM 기반 코딩 에이전트.

    base_url이 없으면 내장 모델(mlx-lm)을 사용.
    base_url이 있으면 외부 API 서버(LM Studio/Ollama)를 사용.
    """

    def __init__(self, model: str = None, base_url: str = None, cwd: str = None,
                 confirm_fn=None):
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url  # None이면 내장 모델 사용
        self.cwd = cwd or os.getcwd()
        self.confirm_fn = confirm_fn
        self.messages = []
        self._streamed = False
        self._local_llm = None  # 내장 모델 인스턴스
        self._init_system_prompt()

    @property
    def use_local(self) -> bool:
        """내장 모델 사용 여부."""
        return self.base_url is None

    def _init_system_prompt(self):
        self.messages = [
            {"role": "system", "content": _build_system_prompt(self.cwd)},
        ]

    def reset(self):
        """대화 히스토리 초기화."""
        self._init_system_prompt()

    def load_model(self, on_status=None):
        """내장 모델 로드 (로컬 모드 전용)."""
        if not self.use_local:
            return

        from backend import LocalLLM
        self._local_llm = LocalLLM(model_id=self.model)
        self._local_llm.load(on_status=on_status)

    def is_model_ready(self) -> bool:
        """모델이 사용 가능한 상태인지 확인."""
        if self.use_local:
            return self._local_llm is not None and self._local_llm.is_loaded
        return self.check_server()

    def chat(self, user_input: str, on_text=None, on_tool=None):
        """사용자 입력을 처리하고 응답을 반환.

        Args:
            user_input: 사용자 텍스트 입력
            on_text: 텍스트 스트리밍 콜백
            on_tool: 도구 사용 콜백
        """
        self.messages.append({"role": "user", "content": user_input})
        return self._run_agent_loop(on_text=on_text, on_tool=on_tool)

    def _run_agent_loop(self, on_text=None, on_tool=None, max_iterations=20):
        """에이전트 루프: 도구 호출이 없을 때까지 반복."""
        for _ in range(max_iterations):
            response_text = self._call_llm(on_text=on_text)

            # 응답에서 도구 호출 파싱
            tool_calls = _parse_tool_calls(response_text)

            if not tool_calls:
                # 도구 호출 없음 → 최종 응답
                clean = re.sub(r"</?tool_call>", "", response_text).strip()
                self.messages.append({"role": "assistant", "content": clean})
                return clean

            # 도구 호출이 있으면 텍스트와 함께 저장
            self.messages.append({"role": "assistant", "content": response_text})

            # 도구 실행
            for call in tool_calls:
                tool_name = call["name"]
                tool_args = call.get("arguments", {})

                # 위험 도구 확인
                if tool_name in DANGEROUS_TOOLS and self.confirm_fn:
                    if not self.confirm_fn(tool_name, tool_args):
                        tool_result = "[거부됨] 사용자가 실행을 거부했습니다."
                        self.messages.append({
                            "role": "user",
                            "content": f"<tool_result>\n{tool_result}\n</tool_result>",
                        })
                        if on_tool:
                            on_tool(tool_name, tool_args, tool_result)
                        continue

                tool_result = execute_tool(tool_name, tool_args, self.cwd)

                self.messages.append({
                    "role": "user",
                    "content": f"<tool_result>\n{tool_result}\n</tool_result>",
                })

                if on_tool:
                    on_tool(tool_name, tool_args, tool_result)

        return "[오류] 최대 반복 횟수 초과"

    def _call_llm(self, on_text=None) -> str:
        """LLM 호출. 내장 모델 또는 외부 API 자동 선택."""
        if self.use_local:
            return self._call_local(on_text=on_text)
        return self._call_api(on_text=on_text)

    def _call_local(self, on_text=None) -> str:
        """내장 모델(mlx-lm)로 추론."""
        self._streamed = False

        if not self._local_llm or not self._local_llm.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        self._streamed = True
        assembled = ""
        repeat_window = ""
        debug = os.environ.get("HAEDONG_DEBUG", "")

        for token_text in self._local_llm.chat_stream(
            messages=self.messages,
            max_tokens=4096,
            temperature=0.2,
        ):
            assembled += token_text
            if on_text:
                on_text(token_text)

            # 반복 감지
            repeat_window += token_text
            if len(repeat_window) > 200:
                repeat_window = repeat_window[-200:]
            if len(repeat_window) >= 90:
                for plen in range(30, len(repeat_window) // 3 + 1):
                    pat = repeat_window[-plen:]
                    if repeat_window.count(pat) >= 3:
                        if debug:
                            print(f"\n[DEBUG] 반복 감지로 중단. 패턴({plen}자): {pat!r}")
                        return assembled

        if debug:
            print(f"\n[DEBUG] 원본 출력:\n{assembled}")

        return assembled

    def _call_api(self, on_text=None) -> str:
        """외부 API 서버(LM Studio/Ollama)로 추론."""
        import requests

        self._streamed = False
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "temperature": 0.2,
            "repeat_penalty": 1.3,
            "max_tokens": 4096,
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"

        self._streamed = True
        assembled = ""
        repeat_window = ""

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                assembled += content
                if on_text:
                    on_text(content)

                repeat_window += content
                if len(repeat_window) > 200:
                    repeat_window = repeat_window[-200:]
                if len(repeat_window) >= 90:
                    for plen in range(30, len(repeat_window) // 3 + 1):
                        pat = repeat_window[-plen:]
                        if repeat_window.count(pat) >= 3:
                            resp.close()
                            return assembled

        return assembled

    def check_server(self) -> bool:
        """외부 API 서버 연결 확인 (API 모드 전용)."""
        if self.use_local:
            return self.is_model_ready()
        import requests
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """사용 가능한 모델 목록."""
        if self.use_local:
            return [self.model]
        import requests
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            return [m.get("id", m.get("name", "unknown")) for m in models]
        except Exception:
            return []
