import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from refiner import refine, scan_project_tree, detect_serena, SYSTEM_PROMPT_BASE


class TestSystemPrompt(unittest.TestCase):
    def test_contains_analysis_instructions(self):
        self.assertIn("프로젝트 소스코드를 분석할 수 있는 도구", SYSTEM_PROMPT_BASE)
        self.assertIn("대상 파일 경로", SYSTEM_PROMPT_BASE)
        self.assertIn("대상 심볼", SYSTEM_PROMPT_BASE)

    def test_requests_plain_text_output(self):
        self.assertIn("개선된 프롬프트만 출력", SYSTEM_PROMPT_BASE)


class TestDetectSerena(unittest.TestCase):
    """detect_serena 테스트. _claude_home을 mock해서 글로벌 설정 영향 차단."""

    def _mock_empty_home(self):
        """글로벌 설정이 없는 가짜 홈 디렉토리."""
        return patch("refiner._claude_home", return_value=tempfile.mkdtemp())

    def test_detects_serena_in_local_mcp_json(self):
        with tempfile.TemporaryDirectory() as tmpdir, self._mock_empty_home():
            mcp = {"mcpServers": {"plugin:serena:serena": {"command": "serena"}}}
            with open(os.path.join(tmpdir, ".mcp.json"), "w") as f:
                json.dump(mcp, f)
            self.assertTrue(detect_serena(tmpdir))

    def test_no_serena_anywhere(self):
        with tempfile.TemporaryDirectory() as tmpdir, self._mock_empty_home():
            mcp = {"mcpServers": {"figma": {"command": "figma"}}}
            with open(os.path.join(tmpdir, ".mcp.json"), "w") as f:
                json.dump(mcp, f)
            self.assertFalse(detect_serena(tmpdir))

    def test_no_config_at_all(self):
        with tempfile.TemporaryDirectory() as tmpdir, self._mock_empty_home():
            self.assertFalse(detect_serena(tmpdir))

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir, self._mock_empty_home():
            with open(os.path.join(tmpdir, ".mcp.json"), "w") as f:
                f.write("invalid json")
            self.assertFalse(detect_serena(tmpdir))

    def test_detects_serena_in_global_plugins(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as fake_home:
            # 글로벌 플러그인에 serena 설정
            plugins_dir = os.path.join(fake_home, "plugins")
            os.makedirs(plugins_dir)
            plugins = {"plugins": {"serena@claude-plugins-official": [{}]}}
            with open(os.path.join(plugins_dir, "installed_plugins.json"), "w") as f:
                json.dump(plugins, f)

            with patch("refiner._claude_home", return_value=fake_home):
                self.assertTrue(detect_serena(tmpdir))

    def test_detects_serena_in_global_mcp(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as fake_home:
            mcp = {"mcpServers": {"serena-server": {"command": "serena"}}}
            with open(os.path.join(fake_home, ".mcp.json"), "w") as f:
                json.dump(mcp, f)

            with patch("refiner._claude_home", return_value=fake_home):
                self.assertTrue(detect_serena(tmpdir))


class TestScanProjectTree(unittest.TestCase):
    def test_scans_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "main.py"), "w").close()
            os.mkdir(os.path.join(tmpdir, "src"))
            open(os.path.join(tmpdir, "src", "app.py"), "w").close()

            tree = scan_project_tree(tmpdir)
            self.assertIn("main.py", tree)
            self.assertIn("src/", tree)

    def test_skips_git_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.mkdir(os.path.join(tmpdir, ".git"))
            open(os.path.join(tmpdir, ".git", "config"), "w").close()
            open(os.path.join(tmpdir, "main.py"), "w").close()

            tree = scan_project_tree(tmpdir)
            self.assertNotIn(".git", tree)

    def test_respects_max_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                open(os.path.join(tmpdir, f"file{i}.py"), "w").close()

            tree = scan_project_tree(tmpdir, max_files=5)
            self.assertIn("more files", tree)


class TestRefineWithMock(unittest.TestCase):
    @patch("refiner.detect_serena", return_value=False)
    @patch("refiner.subprocess.run")
    def test_without_serena(self, mock_run, _):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "개선됨"
        mock_run.return_value = mock_result

        refine("테스트")
        args = mock_run.call_args[0][0]
        self.assertIn("Read", args)
        self.assertNotIn("mcp__plugin_serena_serena__activate_project", args)

    @patch("refiner.detect_serena", return_value=True)
    @patch("refiner.subprocess.run")
    def test_with_serena(self, mock_run, _):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "개선됨"
        mock_run.return_value = mock_result

        refine("테스트")
        args = mock_run.call_args[0][0]
        self.assertIn("mcp__plugin_serena_serena__activate_project", args)
        self.assertIn("mcp__plugin_serena_serena__find_symbol", args)
        self.assertIn("Read", args)

    @patch("refiner.subprocess.run")
    def test_includes_serena_status_in_prompt(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "개선됨"
        mock_run.return_value = mock_result

        refine("테스트")
        prompt_arg = mock_run.call_args[0][0][-1]
        self.assertIn("Serena 사용 가능", prompt_arg)

    @patch("refiner.subprocess.run")
    def test_raises_on_failure(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        mock_run.return_value = mock_result

        with self.assertRaises(RuntimeError):
            refine("테스트")


if __name__ == "__main__":
    unittest.main()
