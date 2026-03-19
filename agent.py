"""해동 코드 - 로컬 LLM 기반 코딩 에이전트 코어.

기본: 내장 모델 (mlx-lm, Apple Silicon 최적화)
옵션: 외부 API 서버 (LM Studio / Ollama) — --url 플래그로 활성화
"""
from __future__ import annotations

import json
import os
import re

from tools import TOOL_SCHEMAS, DANGEROUS_TOOLS, execute_tool
from refiner import scan_project_tree

# --- 기본 설정 ---
DEFAULT_MODEL = "mlx-community/Qwen3-30B-A3B-4bit"

# --- 프롬프트 기반 도구 호출용 시스템 프롬프트 ---

_TOOL_DESCRIPTIONS = """
## 도구 목록

### 읽기/탐색 도구 (안전 — 자유롭게 사용)
1. read_file(path): 파일 내용을 줄 번호와 함께 읽습니다.
2. list_files(pattern, path?): glob 패턴으로 파일 목록을 반환합니다. (예: "**/*.py")
3. search_files(pattern, path?, include?): 파일 내용에서 정규식 검색합니다 (grep).
4. git_status(): Git 상태를 확인합니다.
5. git_diff(path?, staged?): Git diff를 봅니다.
6. git_log(count?): 최근 커밋 로그를 봅니다.

### 수정 도구 (주의 — 반드시 read_file로 확인 후 사용)
7. edit_file(path, old_text, new_text): 파일 내 특정 문자열을 교체합니다.
8. write_file(path, content): 파일을 생성하거나 덮어씁니다.

### 실행 도구 (위험 — 사용자에게 미리 설명)
9. run_command(command): 터미널 명령을 실행합니다.
10. git_commit(message, files?): 변경사항을 커밋합니다.
"""

SYSTEM_PROMPT = """\
당신은 "해동 코드", 로컬 AI 코딩 에이전트입니다.
사용자의 소프트웨어 개발 작업을 도와주는 것이 목표입니다.
항상 한국어로 대답합니다.

## 사고 프로세스 (모든 요청에 반드시 따를 것)

모든 사용자 요청에 대해 다음 단계를 순서대로 수행합니다:

### 1단계: 이해 — 사용자가 정확히 무엇을 원하는지 파악
- 요청이 모호하면 먼저 질문하여 명확히 합니다.
- "파일 수정해줘"처럼 대상이 불분명하면 어떤 파일인지 확인합니다.

### 2단계: 정찰 — 관련 코드를 먼저 탐색
- search_files로 관련 키워드를 검색합니다.
- list_files로 관련 디렉토리 구조를 파악합니다.
- read_file로 수정 대상 파일의 전체 맥락을 이해합니다.
- 절대로 읽지 않은 파일을 수정하지 않습니다.

### 3단계: 계획 — 수정 방향을 사용자에게 설명
- 어떤 파일의 어떤 부분을 왜 바꾸는지 간결하게 설명합니다.
- 파괴적 변경(삭제, 덮어쓰기)은 반드시 사전 고지합니다.

### 4단계: 실행 — 도구를 사용하여 변경
- edit_file 사용 시: old_text는 read_file에서 확인한 정확한 문자열을 사용합니다.
- write_file은 새 파일 생성 시에만 사용합니다. 기존 파일은 edit_file로 수정합니다.

### 5단계: 검증 — 변경 결과 확인
- 수정 후 read_file로 결과를 확인합니다.
- 필요하면 run_command로 테스트나 빌드를 실행합니다.

## 도구 사용 규칙
{tool_descriptions}

도구를 호출하려면 이 형식을 사용하세요:
<tool_call>
{{"name": "도구이름", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

한 번에 하나의 도구만 호출하세요.
도구 실행 결과는 <tool_result> 태그로 전달됩니다.

## 도구 사용 패턴

### 파일 수정 시 (필수 순서)
1. read_file → 대상 파일 내용 확인
2. 사용자에게 변경 계획 설명
3. edit_file → old_text에 정확한 원본 텍스트 사용
4. read_file → 변경 결과 확인

### 버그 수정 시
1. search_files → 에러 메시지나 관련 키워드 검색
2. read_file → 해당 파일 전체 맥락 파악
3. 원인 분석 후 사용자에게 설명
4. edit_file → 수정
5. 관련 테스트가 있으면 run_command로 실행

### 새 기능 추가 시
1. list_files → 프로젝트 구조에서 적절한 위치 탐색
2. read_file → 유사 코드 패턴 확인 (기존 스타일에 맞추기)
3. 구현 계획 설명
4. write_file 또는 edit_file → 코드 작성
5. 검증

## 실패 대응

- edit_file에서 "텍스트를 찾을 수 없습니다" → read_file로 최신 내용을 다시 확인 후 정확한 문자열로 재시도
- search_files에서 "결과 없음" → 다른 키워드나 다른 디렉토리에서 재검색
- run_command 실패 → 에러 메시지를 분석하고 원인과 해결책을 사용자에게 설명
- 3회 이상 같은 작업 실패 → 멈추고 사용자에게 상황 설명, 대안 제시

## 금지 사항

- 읽지 않은 파일을 수정하지 않습니다.
- 사용자가 요청하지 않은 파일을 수정하지 않습니다.
- 추측으로 코드를 작성하지 않습니다. 반드시 현재 코드를 확인 후 작업합니다.
- git commit, git push는 사용자가 명시적으로 요청하지 않는 한 실행하지 않습니다.

## 현재 프로젝트 정보

작업 디렉토리: {cwd}

### 프로젝트 구조
```
{project_tree}
```
"""


def _separate_thinking(text: str) -> tuple:
    """<think> 태그에서 사고 과정과 응답을 분리.

    Returns:
        (thinking_text, response_text) 튜플
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        response = (text[:match.start()] + text[match.end():]).strip()
        return thinking, response
    return "", text


class _ThinkingFilter:
    """스트리밍 중 <think> 태그를 감지하여 사고/응답 콜백을 분리."""

    def __init__(self, on_text=None, on_thinking=None):
        self.on_text = on_text
        self.on_thinking = on_thinking
        self._buf = ""
        self._state = "detect"  # detect → thinking → response
        self._think_buf = ""

    def feed(self, token: str):
        if self._state == "response":
            if self.on_text:
                self.on_text(token)
            return

        self._buf += token

        if self._state == "detect":
            if "<think>" in self._buf:
                self._state = "thinking"
                after = self._buf.split("<think>", 1)[1]
                self._buf = ""
                self._think_buf = after
                if after and self.on_thinking:
                    self.on_thinking(after)
                return
            # < 가 없으면 thinking 없는 것
            if "<" not in self._buf:
                self._state = "response"
                if self.on_text:
                    self.on_text(self._buf)
                self._buf = ""
                return
            # 부분 태그 가능 — 7자 초과하면 thinking 아님
            if len(self._buf) > 7:
                self._state = "response"
                if self.on_text:
                    self.on_text(self._buf)
                self._buf = ""
            return

        # state == "thinking"
        if "</think>" in self._buf:
            self._state = "response"
            parts = self._buf.split("</think>", 1)
            tail = parts[0]
            after = parts[1].lstrip("\n")
            if tail and self.on_thinking:
                self.on_thinking(tail)
            self._buf = ""
            if after and self.on_text:
                self.on_text(after)
        else:
            # </think> 태그가 잘려서 올 수 있으므로 끝 8자는 버퍼에 유지
            safe_len = max(0, len(self._buf) - 8)
            if safe_len > 0:
                to_flush = self._buf[:safe_len]
                self._buf = self._buf[safe_len:]
                if self.on_thinking:
                    self.on_thinking(to_flush)


def _build_system_prompt(cwd: str) -> str:
    tree = scan_project_tree(cwd, max_files=200)
    return SYSTEM_PROMPT.format(
        tool_descriptions=_TOOL_DESCRIPTIONS,
        cwd=cwd,
        project_tree=tree,
    )


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
        self.last_thinking = ""  # 마지막 응답의 사고 과정
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

    def chat(self, user_input: str, on_text=None, on_tool=None, on_thinking=None):
        """사용자 입력을 처리하고 응답을 반환.

        Args:
            user_input: 사용자 텍스트 입력
            on_text: 텍스트 스트리밍 콜백
            on_tool: 도구 사용 콜백
            on_thinking: 사고 과정 스트리밍 콜백
        """
        self.messages.append({"role": "user", "content": user_input})
        return self._run_agent_loop(on_text=on_text, on_tool=on_tool,
                                    on_thinking=on_thinking)

    def _run_agent_loop(self, on_text=None, on_tool=None, on_thinking=None,
                        max_iterations=20):
        """에이전트 루프: 도구 호출이 없을 때까지 반복."""
        for _ in range(max_iterations):
            raw_response = self._call_llm(on_text=on_text,
                                          on_thinking=on_thinking)

            # <think> 태그 분리
            thinking, response_text = _separate_thinking(raw_response)
            if thinking:
                self.last_thinking = thinking

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

    def _call_llm(self, on_text=None, on_thinking=None) -> str:
        """LLM 호출. 내장 모델 또는 외부 API 자동 선택."""
        if self.use_local:
            return self._call_local(on_text=on_text, on_thinking=on_thinking)
        return self._call_api(on_text=on_text, on_thinking=on_thinking)

    def _call_local(self, on_text=None, on_thinking=None) -> str:
        """내장 모델(mlx-lm)로 추론."""
        self._streamed = False

        if not self._local_llm or not self._local_llm.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        self._streamed = True
        assembled = ""
        repeat_window = ""
        debug = os.environ.get("HAEDONG_DEBUG", "")
        filt = _ThinkingFilter(on_text=on_text, on_thinking=on_thinking)

        for token_text in self._local_llm.chat_stream(
            messages=self.messages,
            max_tokens=4096,
            temperature=0.2,
        ):
            assembled += token_text
            filt.feed(token_text)

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

    def _call_api(self, on_text=None, on_thinking=None) -> str:
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
        filt = _ThinkingFilter(on_text=on_text, on_thinking=on_thinking)

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
                filt.feed(content)

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
