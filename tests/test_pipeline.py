from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


AUTOMATION_DIR = Path(__file__).resolve().parents[1] / "automation"
sys.path.insert(0, str(AUTOMATION_DIR))

from pipeline import WorkflowError, _enforce_change_scope  # noqa: E402


class PipelineScopeTests(unittest.TestCase):
    def _repository(self) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
        (root / "source").mkdir()
        (root / "source" / "index.md").write_text("# Initial\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
        return root

    def test_public_source_change_is_allowed(self) -> None:
        root = self._repository()
        (root / "source" / "index.md").write_text("# Updated\n", encoding="utf-8")
        self.assertEqual({"source/index.md"}, _enforce_change_scope(root))

    def test_automation_change_is_blocked(self) -> None:
        root = self._repository()
        (root / "automation").mkdir()
        (root / "automation" / "unsafe.txt").write_text("unsafe\n", encoding="utf-8")
        with self.assertRaises(WorkflowError):
            _enforce_change_scope(root)


if __name__ == "__main__":
    unittest.main()
