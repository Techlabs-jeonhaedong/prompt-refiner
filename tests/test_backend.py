"""backend.py 테스트 — 모델 로딩 없이 인터페이스 검증."""

import unittest
from unittest.mock import patch, MagicMock

from backend import LocalLLM, DEFAULT_MODEL_ID


class TestLocalLLMInit(unittest.TestCase):
    def test_default_model_id(self):
        llm = LocalLLM()
        self.assertEqual(llm.model_id, DEFAULT_MODEL_ID)
        self.assertIn("Qwen3-30B-A3B", llm.model_id)

    def test_custom_model_id(self):
        llm = LocalLLM(model_id="mlx-community/some-other-model")
        self.assertEqual(llm.model_id, "mlx-community/some-other-model")

    def test_not_loaded_initially(self):
        llm = LocalLLM()
        self.assertFalse(llm.is_loaded)
        self.assertIsNone(llm.model)
        self.assertIsNone(llm.tokenizer)

    def test_unload_clears_state(self):
        llm = LocalLLM()
        llm.model = "fake"
        llm.tokenizer = "fake"
        self.assertTrue(llm.is_loaded)

        llm.unload()
        self.assertFalse(llm.is_loaded)
        self.assertIsNone(llm.model)
        self.assertIsNone(llm.tokenizer)


class TestLocalLLMErrors(unittest.TestCase):
    def test_chat_stream_without_load_raises(self):
        llm = LocalLLM()
        with self.assertRaises(RuntimeError) as ctx:
            list(llm.chat_stream([{"role": "user", "content": "hello"}]))
        self.assertIn("로드되지 않았습니다", str(ctx.exception))

    def test_generate_without_load_raises(self):
        llm = LocalLLM()
        with self.assertRaises(RuntimeError) as ctx:
            llm.generate([{"role": "user", "content": "hello"}])
        self.assertIn("로드되지 않았습니다", str(ctx.exception))


class TestLocalLLMLoad(unittest.TestCase):
    @patch("backend.LocalLLM.load")
    def test_load_calls_on_status(self, mock_load):
        """on_status 콜백이 호출되는지 확인 (mock으로)."""
        llm = LocalLLM()
        status_calls = []
        mock_load.side_effect = lambda on_status=None: (
            on_status("테스트") if on_status else None
        )
        llm.load(on_status=lambda msg: status_calls.append(msg))
        self.assertEqual(status_calls, ["테스트"])


if __name__ == "__main__":
    unittest.main()
