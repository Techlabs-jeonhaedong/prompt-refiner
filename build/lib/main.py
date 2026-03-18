#!/usr/bin/env python3
"""프롬프트 개선기 - Claude Code를 활용해 당신의 클로드 코드가 더 명료하게 일하도록 합니다. 변환"""

import json
import random
import sys
import threading
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.live import Live
import os
from refiner import refine, execute

console = Console()

LOGO = r"""
[bold cyan]  ██╗  ██╗ █████╗ ███████╗██████╗  ██████╗ ███╗   ██╗ ██████╗[/]
[bold cyan]  ██║  ██║██╔══██╗██╔════╝██╔══██╗██╔═══██╗████╗  ██║██╔════╝[/]
[bold cyan]  ███████║███████║█████╗  ██║  ██║██║   ██║██╔██╗ ██║██║  ███╗[/]
[bold cyan]  ██╔══██║██╔══██║██╔══╝  ██║  ██║██║   ██║██║╚██╗██║██║   ██║[/]
[bold cyan]  ██║  ██║██║  ██║███████╗██████╔╝╚██████╔╝██║ ╚████║╚██████╔╝[/]
[bold cyan]  ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝[/]
[bold white]             ┄┄┄  해 동  코 드  ┄┄┄[/]"""


def print_intro():
    cwd = os.getcwd()

    console.print()
    console.print(LOGO)
    console.print()
    console.print(
        "  [bold white on cyan]  ✦ 해동 코드  [/]"
        "  [dim]당신의 클로드 코드가 더 명료하게 일하도록 합니다.[/]"
    )
    console.print()
    console.print(f"  [dim]프로젝트[/]  [bold]{os.path.basename(cwd)}/[/]")
    console.print(f"  [dim]종료    [/]  [dim]q · Ctrl+C[/]")
    console.print()


def display(original: str, refined: str):
    orig_len = len(original)
    new_len = len(refined)
    diff = orig_len - new_len

    before = Panel(
        Text(original, style="white"),
        title="[bold red]✗ BEFORE[/]",
        border_style="dim red",
        padding=(1, 2),
        expand=True,
    )

    after = Panel(
        Text(refined, style="white"),
        title="[bold cyan]✓ AFTER[/]",
        border_style="dim cyan",
        padding=(1, 2),
        expand=True,
    )

    console.print()
    console.print(Columns([before, after], equal=True, padding=(0, 1)))

    if diff > 0:
        ratio = diff / orig_len * 100
        console.print(
            f"  [dim]📊 {orig_len}자 → {new_len}자[/]  [bold cyan](-{ratio:.0f}%)[/]",
        )
    else:
        console.print(f"  [dim]📊 {orig_len}자 → {new_len}자[/]")
    console.print()


LOADING_MESSAGES = [
    "✦ 프로젝트 구조를 살펴보는 중...",
    "✦ 소스코드를 꼼꼼히 읽는 중...",
    "✦ 프롬프트를 이 프로젝트에 맞게 맛있게 바꾸는 중...",
    "✦ 모호한 표현을 콕콕 찍어내는 중...",
    "✦ 어떤 파일을 건드려야 할지 추리하는 중...",
    "✦ AI가 바로 실행할 수 있게 다듬는 중...",
    "✦ 거의 다 됐어, 마무리 중...",
]

def stream_output(proc):
    """stream-json 이벤트를 파싱해서 실시간으로 표시."""
    for raw_line in proc.stdout:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        # 텍스트 스트리밍 (토큰 단위)
        if event_type == "stream_event":
            delta = event.get("event", {}).get("delta", {})
            if delta.get("type") == "text_delta":
                sys.stdout.write(delta.get("text", ""))
                sys.stdout.flush()

        # 도구 사용 시작
        elif event_type == "assistant":
            message = event.get("message", {})
            if isinstance(message, dict):
                for block in message.get("content", []):
                    if block.get("type") == "tool_use":
                        tool = block.get("name", "")
                        inp = block.get("input", {})
                        _print_tool_use(tool, inp)

        # 도구 실행 결과
        elif event_type == "tool_result":
            pass  # 결과는 너무 길 수 있으니 생략

        # 최종 결과
        elif event_type == "result":
            console.print()  # 마지막 줄바꿈

    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read()
        if err.strip():
            console.print(f"\n  [bold red]✗ 실행 오류:[/] {err}\n")
    console.print()


def _print_tool_use(tool: str, inp: dict):
    """도구 호출을 한 줄로 표시."""
    if tool == "Read":
        path = inp.get("file_path", "")
        console.print(f"  [dim]📖 Read → {path}[/]")
    elif tool == "Edit":
        path = inp.get("file_path", "")
        console.print(f"  [dim]✏️  Edit → {path}[/]")
    elif tool == "Write":
        path = inp.get("file_path", "")
        console.print(f"  [dim]📝 Write → {path}[/]")
    elif tool == "Bash":
        cmd = inp.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        console.print(f"  [dim]💻 Bash → {cmd}[/]")
    elif tool == "Glob":
        pattern = inp.get("pattern", "")
        console.print(f"  [dim]🔍 Glob → {pattern}[/]")
    elif tool == "Grep":
        pattern = inp.get("pattern", "")
        console.print(f"  [dim]🔍 Grep → {pattern}[/]")
    else:
        console.print(f"  [dim]🔧 {tool}[/]")


def refine_with_progress(original: str) -> str:
    result_holder = {}
    error_holder = {}

    def run():
        try:
            result_holder["value"] = refine(original)
        except Exception as e:
            error_holder["error"] = e

    worker = threading.Thread(target=run, daemon=True)

    messages = LOADING_MESSAGES.copy()
    random.shuffle(messages)

    with Live(console=console, refresh_per_second=10, transient=True) as live:
        worker.start()
        idx = 0
        tick = 0
        while worker.is_alive():
            msg = messages[idx % len(messages)]
            spinner = Spinner("dots", text=f"[bold cyan]  {msg}[/]")
            live.update(spinner)
            worker.join(timeout=10.0)
            tick += 1
            if tick >= 1:
                idx += 1
                tick = 0
        worker.join()

    if "error" in error_holder:
        raise error_holder["error"]

    return result_holder["value"]


def main():
    print_intro()

    while True:
        try:
            original = Prompt.ask("  [bold cyan]❯[/]")
            if original.strip().lower() in ("q", "quit", "exit"):
                console.print("\n  [dim]👋 bye![/]\n")
                break
            if not original.strip():
                continue
            try:
                refined = refine_with_progress(original)
            except Exception as e:
                console.print(f"\n  [bold red]✗ 오류 발생:[/] {e}\n")
                continue
            display(original, refined)

            # Opus로 실행
            console.print("  [bold cyan]⚡ Opus로 실행 중...[/]\n")
            proc = execute(refined)
            try:
                stream_output(proc)
            except KeyboardInterrupt:
                proc.terminate()
                console.print("\n  [dim]실행 중단됨[/]\n")
        except (KeyboardInterrupt, EOFError):
            console.print("\n")
            break


if __name__ == "__main__":
    main()
