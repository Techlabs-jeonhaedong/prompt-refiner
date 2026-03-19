import unittest
import os
import tempfile
import subprocess
from tools import execute_tool, TOOL_SCHEMAS, DANGEROUS_TOOLS, SAFE_TOOLS, _resolve_path


class TestToolSchemas(unittest.TestCase):
    def test_all_schemas_have_required_fields(self):
        for schema in TOOL_SCHEMAS:
            self.assertEqual(schema["type"], "function")
            func = schema["function"]
            self.assertIn("name", func)
            self.assertIn("description", func)
            self.assertIn("parameters", func)

    def test_tool_count(self):
        self.assertEqual(len(TOOL_SCHEMAS), 10)

    def test_dangerous_and_safe_cover_all(self):
        all_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        self.assertEqual(DANGEROUS_TOOLS | SAFE_TOOLS, all_names)


class TestResolvePath(unittest.TestCase):
    def test_relative_path(self):
        result = _resolve_path("src/main.py", "/project")
        self.assertEqual(result, "/project/src/main.py")

    def test_absolute_path(self):
        result = _resolve_path("/tmp/test.py", "/project")
        self.assertEqual(result, "/tmp/test.py")


class TestReadFile(unittest.TestCase):
    def test_reads_file_with_line_numbers(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\nworld\n")
            path = f.name
        try:
            result = execute_tool("read_file", {"path": path}, os.path.dirname(path))
            self.assertIn("1", result)
            self.assertIn("hello", result)
            self.assertIn("world", result)
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        result = execute_tool("read_file", {"path": "/nonexistent/file.txt"}, "/tmp")
        self.assertIn("오류", result)

    def test_directory_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = execute_tool("read_file", {"path": tmpdir}, "/tmp")
            self.assertIn("디렉토리", result)


class TestWriteFile(unittest.TestCase):
    def test_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            result = execute_tool("write_file", {"path": "test.txt", "content": "hello"}, tmpdir)
            self.assertIn("완료", result)
            with open(path) as f:
                self.assertEqual(f.read(), "hello")

    def test_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = execute_tool(
                "write_file",
                {"path": "sub/dir/test.txt", "content": "nested"},
                tmpdir,
            )
            self.assertIn("완료", result)
            with open(os.path.join(tmpdir, "sub", "dir", "test.txt")) as f:
                self.assertEqual(f.read(), "nested")


class TestEditFile(unittest.TestCase):
    def test_edits_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.py")
            with open(path, "w") as f:
                f.write("def foo():\n    return 1\n")

            result = execute_tool(
                "edit_file",
                {"path": "test.py", "old_text": "return 1", "new_text": "return 2"},
                tmpdir,
            )
            self.assertIn("완료", result)
            with open(path) as f:
                self.assertIn("return 2", f.read())

    def test_text_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.py")
            with open(path, "w") as f:
                f.write("hello")

            result = execute_tool(
                "edit_file",
                {"path": "test.py", "old_text": "nonexistent", "new_text": "x"},
                tmpdir,
            )
            self.assertIn("오류", result)


class TestListFiles(unittest.TestCase):
    def test_lists_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.py"), "w").close()
            open(os.path.join(tmpdir, "b.py"), "w").close()
            open(os.path.join(tmpdir, "c.txt"), "w").close()

            result = execute_tool("list_files", {"pattern": "*.py"}, tmpdir)
            self.assertIn("a.py", result)
            self.assertIn("b.py", result)
            self.assertNotIn("c.txt", result)

    def test_no_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = execute_tool("list_files", {"pattern": "*.xyz"}, tmpdir)
            self.assertIn("없음", result)


class TestSearchFiles(unittest.TestCase):
    def test_searches_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("def hello():\n    print('world')\n")

            result = execute_tool("search_files", {"pattern": "hello", "path": tmpdir}, tmpdir)
            self.assertIn("hello", result)

    def test_no_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("nothing here")

            result = execute_tool("search_files", {"pattern": "xyz123", "path": tmpdir}, tmpdir)
            self.assertIn("없음", result)


class TestRunCommand(unittest.TestCase):
    def test_runs_command(self):
        result = execute_tool("run_command", {"command": "echo hello"}, "/tmp")
        self.assertEqual(result, "hello")

    def test_captures_exit_code(self):
        result = execute_tool("run_command", {"command": "exit 1"}, "/tmp")
        self.assertIn("exit code: 1", result)


class TestGitTools(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        subprocess.run(["git", "init", "-q"], cwd=self.tmpdir, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=self.tmpdir, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.tmpdir, check=True)
        with open(os.path.join(self.tmpdir, "file.txt"), "w") as f:
            f.write("initial")
        subprocess.run(["git", "add", "-A"], cwd=self.tmpdir, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=self.tmpdir, check=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_git_status(self):
        result = execute_tool("git_status", {}, self.tmpdir)
        self.assertIn("변경사항 없음", result)

    def test_git_status_with_changes(self):
        with open(os.path.join(self.tmpdir, "new.txt"), "w") as f:
            f.write("new file")
        result = execute_tool("git_status", {}, self.tmpdir)
        self.assertIn("new.txt", result)

    def test_git_log(self):
        result = execute_tool("git_log", {"count": 5}, self.tmpdir)
        self.assertIn("init", result)

    def test_git_diff(self):
        with open(os.path.join(self.tmpdir, "file.txt"), "w") as f:
            f.write("modified")
        result = execute_tool("git_diff", {}, self.tmpdir)
        self.assertIn("modified", result)

    def test_git_commit(self):
        with open(os.path.join(self.tmpdir, "new.txt"), "w") as f:
            f.write("new")
        result = execute_tool("git_commit", {"message": "add new", "files": ["new.txt"]}, self.tmpdir)
        self.assertIn("add new", result)


class TestUnknownTool(unittest.TestCase):
    def test_unknown_tool(self):
        result = execute_tool("nonexistent", {}, "/tmp")
        self.assertIn("오류", result)


if __name__ == "__main__":
    unittest.main()
