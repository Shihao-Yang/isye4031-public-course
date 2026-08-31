from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


AUTOMATION_DIR = Path(__file__).resolve().parents[1] / "automation"
sys.path.insert(0, str(AUTOMATION_DIR))

from privacy import (  # noqa: E402
    html_to_text,
    normalize_protected_terms,
    redact_text,
    scan_file,
    scan_text,
    scan_tree,
)


class PrivacyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.terms = normalize_protected_terms(
            ["Example Student", "Student, Example", "987654", "测试同学", "Li"]
        )

    def test_html_extraction_drops_active_content(self) -> None:
        value = "<h1>Lesson</h1><script>steal()</script><p>Regression review</p>"
        extracted = html_to_text(value)
        self.assertIn("Lesson", extracted)
        self.assertIn("Regression review", extracted)
        self.assertNotIn("steal", extracted)

    def test_redaction_removes_common_sensitive_values(self) -> None:
        fake_token = "2096~" + "abcdefghijklmnopqrstuvwxyz123456"
        value = (
            "Example Student wrote student@example.edu. "
            "See https://gatech.instructure.com/courses/123/pages/private. "
            f"Authorization: Bearer {fake_token}."
        )
        redacted = redact_text(value, self.terms)
        self.assertNotIn("Example Student", redacted)
        self.assertNotIn("student@example.edu", redacted)
        self.assertNotIn("instructure.com", redacted)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", redacted)

    def test_scan_reports_categories_without_sensitive_values(self) -> None:
        findings = scan_text(
            "Example Student can be reached at student@example.edu.",
            self.terms,
            "source/example.md",
        )
        rendered = " ".join(f"{item.category} {item.path}" for item in findings)
        self.assertIn("email-address", rendered)
        self.assertIn("protected-student-or-course-identifier", rendered)
        self.assertNotIn("Example Student", rendered)
        self.assertNotIn("student@example.edu", rendered)

    def test_scan_tree_passes_clean_public_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            (source / "index.md").write_text(
                "# Regression and Forecasting\n\nA public lesson about least squares.\n",
                encoding="utf-8",
            )
            self.assertEqual([], scan_tree(root, self.terms, require_binary_tools=False))

    def test_short_ambiguous_latin_name_is_not_a_protected_term(self) -> None:
        self.assertNotIn("Li", self.terms)
        self.assertIn("测试同学", self.terms)

    def test_raw_lms_identifier_in_filename_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "canvas-12345-slides.pdf"
            path.write_bytes(b"")
            findings = scan_file(path, self.terms, display_path=path.name)
            self.assertEqual("raw-lms-identifier-in-filename", findings[0].category)


if __name__ == "__main__":
    unittest.main()
