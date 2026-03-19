"""해동 코드 - 로컬 LLM 추론 백엔드 (Apple Silicon MLX).

mlx-lm 라이브러리를 사용하여 텍스트 생성 모델을 로컬에서 실행.
외부 서버(LM Studio, Ollama) 없이 프로그램 단독으로 동작.
"""
from __future__ import annotations

import gc

# 기본 모델: Qwen3-30B-A3B 4bit 양자화 (Apple Silicon MoE 최적화)
DEFAULT_MODEL_ID = "mlx-community/Qwen3-30B-A3B-4bit"


class LocalLLM:
    """Apple Silicon 최적화 로컬 LLM 엔진."""

    def __init__(self, model_id: str = None):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.model = None
        self.tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, on_status=None):
        """모델을 메모리에 로드. 최초 실행 시 HuggingFace에서 자동 다운로드.

        Args:
            on_status: 상태 메시지 콜백 fn(str)
        """
        from mlx_lm import load

        if on_status:
            on_status(f"모델 로딩 중: {self.model_id}")

        self.model, self.tokenizer = load(self.model_id)

        if on_status:
            on_status("모델 준비 완료")

    def chat_stream(self, messages: list[dict],
                    max_tokens: int = 4096, temperature: float = 0.2):
        """스트리밍 채팅 생성.

        Args:
            messages: OpenAI 형식 메시지 리스트 [{"role": ..., "content": ...}]
            max_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도 (0.0 = 결정적, 1.0 = 창의적)

        Yields:
            생성된 텍스트 토큰 (str)
        """
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. load()를 먼저 호출하세요.")

        from mlx_lm import stream_generate

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        for response in stream_generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
        ):
            yield response.text

    def generate(self, messages: list[dict],
                 max_tokens: int = 4096, temperature: float = 0.2) -> str:
        """텍스트 생성 (논스트리밍).

        Returns:
            생성된 전체 텍스트
        """
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        return generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )

    def unload(self):
        """모델 메모리 해제."""
        self.model = None
        self.tokenizer = None
        gc.collect()
