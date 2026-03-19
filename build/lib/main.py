#!/usr/bin/env python3
"""프롬프트 개선기 - Claude Code를 활용해 당신의 클로드 코드가 더 명료하게 일하도록 합니다. 변환"""

import random
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

            # Opus 대화형 실행 — 터미널을 Claude에게 넘김
            console.print("\n  [bold cyan]⚡ Opus 대화형 세션 시작[/]\n")
            returncode = execute(refined)
            if returncode != 0:
                console.print(f"\n  [bold red]✗ Claude 종료 코드: {returncode}[/]\n")
        except (KeyboardInterrupt, EOFError):
            console.print("\n")
            break


if __name__ == "__main__":
    main()
