#!/usr/bin/env python3
"""해동 코드 - 로컬 VL 모델 기반 코딩 에이전트.

기본: 내장 Qwen3-VL 모델 (Apple Silicon MLX)
옵션: --url 플래그로 외부 API 서버 사용
"""

import os
import sys
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from agent import Agent

console = Console()

LOGO = r"""
[bold cyan]  ██╗  ██╗ █████╗ ███████╗██████╗  ██████╗ ███╗   ██╗ ██████╗[/]
[bold cyan]  ██║  ██║██╔══██╗██╔════╝██╔══██╗██╔═══██╗████╗  ██║██╔════╝[/]
[bold cyan]  ███████║███████║█████╗  ██║  ██║██║   ██║██╔██╗ ██║██║  ███╗[/]
[bold cyan]  ██╔══██║██╔══██║██╔══╝  ██║  ██║██║   ██║██║╚██╗██║██║   ██║[/]
[bold cyan]  ██║  ██║██║  ██║███████╗██████╔╝╚██████╔╝██║ ╚████║╚██████╔╝[/]
[bold cyan]  ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝[/]
[bold white]             ┄┄┄  해 동  코 드  ┄┄┄[/]"""

def print_intro(model: str, mode: str):
    cwd = os.getcwd()
    console.print()
    console.print(LOGO)
    console.print()
    console.print(
        "  [bold white on cyan]  ✦ 해동 코드  [/]"
        "  [dim]로컬 코딩 에이전트[/]"
    )
    console.print()
    console.print(f"  [dim]모드    [/]  [bold]{mode}[/]")
    console.print(f"  [dim]모델    [/]  [bold]{model}[/]")
    console.print(f"  [dim]프로젝트[/]  [bold]{os.path.basename(cwd)}/[/]")
    console.print(f"  [dim]종료    [/]  [dim]q · Ctrl+C[/]")
    console.print(f"  [dim]초기화  [/]  [dim]/reset[/]")
    console.print(f"  [dim]사고확인[/]  [dim]/think[/]")
    console.print()


def confirm_tool(tool_name: str, args: dict) -> bool:
    """위험 도구 실행 전 사용자 확인."""
    console.print()

    if tool_name == "write_file":
        desc = f"  [bold yellow]⚠ 파일 쓰기:[/] {args.get('path', '?')}"
    elif tool_name == "edit_file":
        desc = f"  [bold yellow]⚠ 파일 수정:[/] {args.get('path', '?')}"
    elif tool_name == "run_command":
        desc = f"  [bold yellow]⚠ 명령 실행:[/] {args.get('command', '?')}"
    elif tool_name == "git_commit":
        desc = f"  [bold yellow]⚠ Git 커밋:[/] {args.get('message', '?')}"
    else:
        desc = f"  [bold yellow]⚠ {tool_name}[/]"

    console.print(desc)
    try:
        answer = Prompt.ask("  [dim]실행할까요?[/]", choices=["y", "n"], default="y")
        return answer == "y"
    except (KeyboardInterrupt, EOFError):
        return False


def on_tool_use(tool_name: str, args: dict, result: str):
    """도구 사용 시 표시."""
    icon_map = {
        "read_file": "📖", "write_file": "✏️", "edit_file": "🔧",
        "list_files": "📁", "search_files": "🔍", "run_command": "💻",
        "git_status": "📊", "git_diff": "📝", "git_commit": "✅", "git_log": "📜",
    }
    icon = icon_map.get(tool_name, "🔧")

    if tool_name in ("read_file", "write_file", "edit_file"):
        detail = args.get("path", "")
    elif tool_name in ("list_files", "search_files"):
        detail = args.get("pattern", "")
    elif tool_name == "run_command":
        detail = args.get("command", "")
    elif tool_name == "git_commit":
        detail = args.get("message", "")
    else:
        detail = ""

    console.print(f"  [dim]{icon} {tool_name}[/] [dim italic]{detail}[/]")

    if result.startswith("[거부됨]"):
        console.print(f"  [dim red]{result}[/]")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="해동 코드 - 로컬 VL 코딩 에이전트")
    parser.add_argument("--model", "-m", default=None, help="모델 ID (HuggingFace)")
    parser.add_argument("--url", "-u", default=None,
                        help="외부 API 서버 URL (미지정 시 내장 모델 사용)")
    parser.add_argument("prompt", nargs="*", help="초기 프롬프트 (선택)")
    args = parser.parse_args()

    agent = Agent(model=args.model, base_url=args.url, confirm_fn=confirm_tool)

    if agent.use_local:
        # 내장 모델 모드: 모델 로드
        console.print()
        console.print("  [bold cyan]⟳[/] [dim]내장 모델을 준비하는 중...[/]")
        console.print(f"  [dim]모델: {agent.model}[/]")
        console.print(f"  [dim]최초 실행 시 모델 다운로드가 필요합니다 (~18GB)[/]")
        console.print()

        try:
            with console.status("  [bold cyan]모델 로딩 중...[/]", spinner="dots"):
                agent.load_model(
                    on_status=lambda msg: None  # 스피너가 상태 표시
                )
        except ImportError:
            console.print(
                "\n  [bold red]✗ mlx-lm이 설치되지 않았습니다.[/]"
                "\n  [dim]pip install mlx-lm 으로 설치하세요.[/]"
                "\n  [dim]또는 --url 플래그로 외부 API 서버를 사용하세요.[/]\n"
            )
            sys.exit(1)
        except Exception as e:
            console.print(f"\n  [bold red]✗ 모델 로딩 실패:[/] {e}\n")
            sys.exit(1)

        mode = "내장 모델 (MLX)"
    else:
        # 외부 API 모드: 서버 연결 확인
        if not agent.check_server():
            console.print(
                f"\n  [bold red]✗ API 서버에 연결할 수 없습니다.[/]"
                f"\n  [dim]서버 주소: {agent.base_url}[/]"
                f"\n  [dim]서버를 시작하거나 --url 없이 내장 모델을 사용하세요.[/]\n"
            )
            sys.exit(1)

        # 외부 API 모드에서 모델 자동 선택
        if not args.model:
            models = agent.list_models()
            if models:
                agent.model = models[0]

        mode = f"외부 API ({agent.base_url})"

    print_intro(agent.model, mode)

    # 초기 프롬프트
    initial_prompt = " ".join(args.prompt) if args.prompt else None

    while True:
        try:
            if initial_prompt:
                user_input = initial_prompt
                console.print(f"  [bold cyan]❯[/] {user_input}")
                initial_prompt = None
            else:
                user_input = Prompt.ask("  [bold cyan]❯[/]")

            stripped = user_input.strip().lower()
            if stripped in ("q", "quit", "exit"):
                console.print("\n  [dim]👋 bye![/]\n")
                break
            if stripped == "/reset":
                agent.reset()
                console.print("  [dim]대화가 초기화되었습니다.[/]\n")
                continue
            if stripped == "/think":
                if agent.last_thinking:
                    console.print(Panel(
                        Text(agent.last_thinking, style="dim"),
                        title="[bold]💭 사고 과정[/]",
                        border_style="dim",
                        expand=False,
                        padding=(1, 2),
                    ))
                else:
                    console.print("  [dim]표시할 사고 과정이 없습니다.[/]")
                console.print()
                continue
            if stripped == "/model":
                models = agent.list_models()
                console.print(f"  [dim]현재 모델:[/] [bold]{agent.model}[/]")
                if models:
                    console.print(f"  [dim]사용 가능:[/] {', '.join(models)}")
                console.print()
                continue
            if stripped.startswith("/model "):
                new_model = user_input.strip()[7:].strip()
                agent.model = new_model
                agent.reset()
                if agent.use_local:
                    console.print(f"  [dim]모델 변경 중...[/]")
                    try:
                        with console.status("  [bold cyan]모델 로딩 중...[/]", spinner="dots"):
                            agent.load_model()
                    except Exception as e:
                        console.print(f"  [bold red]✗ 모델 로딩 실패:[/] {e}\n")
                        continue
                console.print(f"  [dim]모델 변경:[/] [bold]{new_model}[/]\n")
                continue
            if not user_input.strip():
                continue

            console.print()

            # 스트리밍: Live 디스플레이로 마크다운 실시간 렌더링
            response_chunks = []
            thinking_line_count = [0]

            with Live(console=console, refresh_per_second=8, vertical_overflow="visible") as live:
                def on_thinking(chunk):
                    thinking_line_count[0] += chunk.count("\n")
                    live.update(Text(
                        f"  💭 사고 중... ({thinking_line_count[0] + 1}줄)",
                        style="dim italic",
                    ))

                def on_text(chunk):
                    response_chunks.append(chunk)
                    accumulated = "".join(response_chunks)
                    try:
                        live.update(Markdown(accumulated))
                    except Exception:
                        live.update(accumulated)

                response = agent.chat(
                    user_input.strip(),
                    on_text=on_text,
                    on_tool=on_tool_use,
                    on_thinking=on_thinking,
                )

            # 스트리밍이 아닌 경우 최종 마크다운 출력
            if not agent._streamed:
                console.print(Markdown(response))

            # 사고 과정 요약 표시
            if agent.last_thinking and thinking_line_count[0] > 0:
                lines = agent.last_thinking.count("\n") + 1
                console.print(
                    f"  [dim]💭 사고 과정 ({lines}줄)"
                    f" — /think 입력으로 확인[/]"
                )

            console.print()

        except KeyboardInterrupt:
            console.print("\n")
            continue
        except EOFError:
            console.print("\n  [dim]👋 bye![/]\n")
            break
        except Exception as e:
            console.print(f"\n  [bold red]✗ 오류:[/] {e}\n")


if __name__ == "__main__":
    main()
