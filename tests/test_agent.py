import unittest
from unittest.mock import patch, MagicMock
from agent import Agent, DEFAULT_MODEL, _parse_tool_calls, _separate_thinking, _ThinkingFilter


class TestParseToolCalls(unittest.TestCase):
    def test_parses_single_tool_call(self):
        text = '내용을 확인할게요.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "main.py"}}\n</tool_call>'
        calls = _parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_file")
        self.assertEqual(calls[0]["arguments"]["path"], "main.py")

    def test_no_tool_call(self):
        text = "그냥 텍스트 응답입니다."
        calls = _parse_tool_calls(text)
        self.assertEqual(calls, [])

    def test_invalid_json_in_tag(self):
        text = "<tool_call>\nnot json\n</tool_call>"
        calls = _parse_tool_calls(text)
        self.assertEqual(calls, [])

    def test_missing_name_field(self):
        text = '<tool_call>\n{"arguments": {"path": "x"}}\n</tool_call>'
        calls = _parse_tool_calls(text)
        self.assertEqual(calls, [])


class TestAgentInit(unittest.TestCase):
    def test_default_values_local_mode(self):
        """기본값: 내장 모델 모드 (base_url=None)."""
        agent = Agent()
        self.assertEqual(agent.model, DEFAULT_MODEL)
        self.assertIsNone(agent.base_url)
        self.assertTrue(agent.use_local)
        self.assertEqual(len(agent.messages), 1)
        self.assertEqual(agent.messages[0]["role"], "system")

    def test_api_mode_with_url(self):
        """--url 지정 시 API 모드."""
        agent = Agent(base_url="http://localhost:1234/v1")
        self.assertFalse(agent.use_local)
        self.assertEqual(agent.base_url, "http://localhost:1234/v1")

    def test_custom_model(self):
        agent = Agent(model="some-model")
        self.assertEqual(agent.model, "some-model")

    def test_reset(self):
        agent = Agent()
        agent.messages.append({"role": "user", "content": "hello"})
        self.assertEqual(len(agent.messages), 2)
        agent.reset()
        self.assertEqual(len(agent.messages), 1)



def _mock_streaming_response(text_chunks):
    """스트리밍 SSE 응답을 시뮬레이션하는 mock 생성."""
    lines = []
    for chunk in text_chunks:
        data = {
            "choices": [{"delta": {"content": chunk}}],
        }
        import json
        lines.append(f"data: {json.dumps(data)}")
    lines.append("data: [DONE]")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.close = MagicMock()
    return mock_resp


class TestAgentChatAPI(unittest.TestCase):
    """외부 API 모드 테스트."""

    @patch("requests.post")
    def test_simple_text_response(self, mock_post):
        mock_post.return_value = _mock_streaming_response(["안녕", "하세요!"])

        agent = Agent(base_url="http://localhost:1234/v1")
        result = agent.chat("안녕")
        self.assertEqual(result, "안녕하세요!")

    @patch("agent.execute_tool")
    @patch("requests.post")
    def test_tool_call_and_response(self, mock_post, mock_exec):
        tool_resp = _mock_streaming_response([
            '파일을 확인할게요.\n<tool_call>\n{"name": "read_file", ',
            '"arguments": {"path": "main.py"}}\n</tool_call>',
        ])
        final_resp = _mock_streaming_response(["파일을 읽었습니다."])

        mock_post.side_effect = [tool_resp, final_resp]
        mock_exec.return_value = "file content here"

        agent = Agent(base_url="http://localhost:1234/v1")
        result = agent.chat("main.py 읽어줘")
        self.assertEqual(result, "파일을 읽었습니다.")
        mock_exec.assert_called_once_with("read_file", {"path": "main.py"}, agent.cwd)

    @patch("agent.execute_tool")
    @patch("requests.post")
    def test_dangerous_tool_confirmed(self, mock_post, mock_exec):
        tool_resp = _mock_streaming_response([
            '<tool_call>\n{"name": "write_file", "arguments": {"path": "test.txt", "content": "hello"}}\n</tool_call>',
        ])
        final_resp = _mock_streaming_response(["작성 완료"])

        mock_post.side_effect = [tool_resp, final_resp]
        mock_exec.return_value = "파일 작성 완료: test.txt"

        confirm = MagicMock(return_value=True)
        agent = Agent(base_url="http://localhost:1234/v1", confirm_fn=confirm)
        result = agent.chat("test.txt 만들어줘")

        confirm.assert_called_once()
        mock_exec.assert_called_once()
        self.assertEqual(result, "작성 완료")

    @patch("agent.execute_tool")
    @patch("requests.post")
    def test_dangerous_tool_denied(self, mock_post, mock_exec):
        tool_resp = _mock_streaming_response([
            '<tool_call>\n{"name": "run_command", "arguments": {"command": "rm -rf /"}}\n</tool_call>',
        ])
        final_resp = _mock_streaming_response(["실행이 거부되었습니다."])

        mock_post.side_effect = [tool_resp, final_resp]

        confirm = MagicMock(return_value=False)
        agent = Agent(base_url="http://localhost:1234/v1", confirm_fn=confirm)
        result = agent.chat("전부 삭제해")

        confirm.assert_called_once()
        mock_exec.assert_not_called()


class TestAgentChatLocal(unittest.TestCase):
    """내장 모델 모드 테스트."""

    def test_chat_without_model_loaded_raises(self):
        """모델 로드 안 한 상태에서 chat 시 에러."""
        agent = Agent()  # local mode, but no model loaded
        with self.assertRaises(RuntimeError):
            agent.chat("안녕")

    @patch("backend.LocalLLM")
    def test_local_chat_stream(self, MockLLM):
        """내장 모델 스트리밍 응답."""
        mock_llm = MagicMock()
        mock_llm.is_loaded = True
        mock_llm.chat_stream.return_value = iter(["안녕", "하세요!"])
        MockLLM.return_value = mock_llm

        agent = Agent()
        agent._local_llm = mock_llm

        texts = []
        result = agent.chat("안녕", on_text=lambda t: texts.append(t))

        self.assertEqual(result, "안녕하세요!")
        self.assertEqual(texts, ["안녕", "하세요!"])


class TestAgentConnection(unittest.TestCase):
    def test_check_server_local_mode(self):
        """로컬 모드에서 모델 미로드 시 False."""
        agent = Agent()
        self.assertFalse(agent.check_server())

    @patch("requests.get")
    def test_check_server_api_success(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        agent = Agent(base_url="http://localhost:1234/v1")
        self.assertTrue(agent.check_server())

    @patch("requests.get")
    def test_check_server_api_failure(self, mock_get):
        from requests import ConnectionError
        mock_get.side_effect = ConnectionError()
        agent = Agent(base_url="http://localhost:1234/v1")
        self.assertFalse(agent.check_server())

    def test_list_models_local(self):
        """로컬 모드에서 모델 목록은 현재 모델만."""
        agent = Agent()
        models = agent.list_models()
        self.assertEqual(models, [agent.model])

    @patch("requests.get")
    def test_list_models_api(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "qwen3-vl-8b"},
                {"id": "gpt-oss-20b"},
            ]
        }
        mock_get.return_value = mock_resp

        agent = Agent(base_url="http://localhost:1234/v1")
        models = agent.list_models()
        self.assertEqual(models, ["qwen3-vl-8b", "gpt-oss-20b"])


class TestSeparateThinking(unittest.TestCase):
    """<think> 태그 분리 테스트."""

    def test_with_thinking(self):
        text = "<think>\n이건 사고 과정\n</think>\n실제 응답"
        thinking, response = _separate_thinking(text)
        self.assertEqual(thinking, "이건 사고 과정")
        self.assertEqual(response, "실제 응답")

    def test_without_thinking(self):
        text = "사고 과정 없는 응답"
        thinking, response = _separate_thinking(text)
        self.assertEqual(thinking, "")
        self.assertEqual(response, "사고 과정 없는 응답")

    def test_empty_thinking(self):
        text = "<think></think>응답만"
        thinking, response = _separate_thinking(text)
        self.assertEqual(thinking, "")
        self.assertEqual(response, "응답만")

    def test_multiline_thinking(self):
        text = "<think>\nline1\nline2\nline3\n</think>\n결과"
        thinking, response = _separate_thinking(text)
        self.assertIn("line1", thinking)
        self.assertIn("line3", thinking)
        self.assertEqual(response, "결과")


class TestThinkingFilter(unittest.TestCase):
    """스트리밍 thinking 필터 테스트."""

    def test_thinking_then_response(self):
        """<think>...</think> 후 응답이 오는 정상 케이스."""
        text_chunks = []
        think_chunks = []

        filt = _ThinkingFilter(
            on_text=lambda t: text_chunks.append(t),
            on_thinking=lambda t: think_chunks.append(t),
        )

        for token in ["<think>", "사고 ", "과정", "</think>", "\n응답 ", "텍스트"]:
            filt.feed(token)

        self.assertTrue(len(think_chunks) > 0)
        self.assertIn("응답", "".join(text_chunks))
        self.assertNotIn("<think>", "".join(text_chunks))

    def test_no_thinking(self):
        """thinking 없이 바로 응답."""
        text_chunks = []
        think_chunks = []

        filt = _ThinkingFilter(
            on_text=lambda t: text_chunks.append(t),
            on_thinking=lambda t: think_chunks.append(t),
        )

        for token in ["안녕", "하세요", "!!"]:
            filt.feed(token)

        self.assertEqual(think_chunks, [])
        self.assertEqual("".join(text_chunks), "안녕하세요!!")

    def test_split_tag_across_tokens(self):
        """<think> 태그가 토큰 경계에서 분리되는 경우."""
        text_chunks = []
        think_chunks = []

        filt = _ThinkingFilter(
            on_text=lambda t: text_chunks.append(t),
            on_thinking=lambda t: think_chunks.append(t),
        )

        for token in ["<", "think>", "생각중", "</", "think>", "결과"]:
            filt.feed(token)

        self.assertTrue(len(think_chunks) > 0)
        self.assertIn("결과", "".join(text_chunks))

    def test_thinking_stored_in_agent(self):
        """Agent.last_thinking에 사고 과정이 저장되는지 확인."""
        agent = Agent(base_url="http://localhost:1234/v1")

        thinking_resp = _mock_streaming_response([
            "<think>\n분석 중...\n</think>\n",
            "최종 답변입니다.",
        ])
        mock_post = MagicMock()
        mock_post.return_value = thinking_resp

        with patch("requests.post", mock_post):
            result = agent.chat("테스트")

        self.assertEqual(result, "최종 답변입니다.")
        self.assertIn("분석 중", agent.last_thinking)


if __name__ == "__main__":
    unittest.main()
