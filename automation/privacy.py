from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Sequence


TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".htm",
    ".html",
    ".js",
    ".json",
    ".md",
    ".svg",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
OFFICE_SUFFIXES = {".docx", ".pptx", ".xlsx"}
PUBLIC_BINARY_SUFFIXES = IMAGE_SUFFIXES | OFFICE_SUFFIXES | {".ico", ".pdf"}

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(
    r"(?x)(?<!\d)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}(?!\d)"
)
CANVAS_URL_RE = re.compile(
    r"(?i)https?://(?:[^\s/]+\.)?(?:instructure\.com|canvas\.gatech\.edu)(?:/[^\s<>'\"]*)?"
)
PRIVATE_COURSE_PATH_RE = re.compile(
    r"(?i)(?:/api/v1/|/courses/\d+|/users/\d+|verifier=|access_token=)"
)
CREDENTIAL_RE = re.compile(
    r"(?i)(?:authorization\s*:\s*bearer\s+\S+|\b\d+~[A-Za-z0-9_-]{20,}\b|"
    r"(?:api|access)[_-]?(?:key|token)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{20,})"
)
REDACTION_MARKER_RE = re.compile(r"(?i)\[redacted(?: [^\]]+)?\]")
RAW_LMS_FILENAME_RE = re.compile(r"(?i)(?:canvas|course|user|file)[-_]?\d{3,}")
ASSESSMENT_PAGE_RE = re.compile(
    r"(?i)\b(?:homework|assignment|submission|graded|answer key|solution key|exam answers?)\b"
)


@dataclass(frozen=True)
class Finding:
    category: str
    path: str


class VisibleTextParser(HTMLParser):
    """Extract readable text while dropping active and embedded content."""

    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }
    DROP_TAGS = {"canvas", "form", "iframe", "noscript", "script", "style", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._drop_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.DROP_TAGS:
            self._drop_depth += 1
            return
        if not self._drop_depth and tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.DROP_TAGS:
            self._drop_depth = max(0, self._drop_depth - 1)
            return
        if not self._drop_depth and tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._drop_depth:
            self._parts.append(data)

    def text(self) -> str:
        raw = html.unescape("".join(self._parts)).replace("\r", "\n")
        lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in raw.splitlines()]
        out: list[str] = []
        for line in lines:
            if line:
                out.append(line)
            elif out and out[-1] != "":
                out.append("")
        return "\n".join(out).strip()


def html_to_text(value: str) -> str:
    parser = VisibleTextParser()
    parser.feed(value)
    parser.close()
    return parser.text()


def _term_is_specific(term: str) -> bool:
    clean = " ".join(term.split()).strip()
    if not clean:
        return False
    if clean.isdigit():
        return len(clean) >= 5
    if re.search(r"[\u3400-\u9fff]", clean):
        return len(clean) >= 2
    return len(clean) >= 4


def normalize_protected_terms(values: Iterable[object]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values:
        if value is None:
            continue
        term = " ".join(str(value).split()).strip()
        if _term_is_specific(term):
            normalized.add(term)
    return tuple(sorted(normalized, key=lambda value: (-len(value), value.casefold())))


def redact_text(value: str, protected_terms: Sequence[str]) -> str:
    text = CANVAS_URL_RE.sub("[redacted private link]", value)
    text = EMAIL_RE.sub("[redacted contact]", text)
    text = PHONE_RE.sub("[redacted contact]", text)
    text = CREDENTIAL_RE.sub("[redacted credential]", text)
    for term in normalize_protected_terms(protected_terms):
        text = re.sub(re.escape(term), "[redacted student information]", text, flags=re.IGNORECASE)
    return text


def _run_text_command(command: list[str], timeout: int = 180) -> str:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.decode("utf-8", errors="replace")


def _ocr_image(path: Path) -> str:
    if not shutil.which("tesseract"):
        return ""
    return _run_text_command(["tesseract", str(path), "stdout"], timeout=240)


def _extract_pdf_text(path: Path, require_tools: bool) -> tuple[str, list[Finding]]:
    findings: list[Finding] = []
    if not shutil.which("pdftotext") or not shutil.which("pdfinfo"):
        if require_tools:
            findings.append(Finding("pdf-inspection-tool-unavailable", path.name))
        return "", findings

    text = _run_text_command(["pdftotext", str(path), "-"])
    metadata = _run_text_command(["pdfinfo", str(path)])

    if shutil.which("pdftoppm") and shutil.which("tesseract"):
        with tempfile.TemporaryDirectory(prefix="canvas-pdf-ocr-") as temp_dir:
            prefix = Path(temp_dir) / "page"
            completed = subprocess.run(
                ["pdftoppm", "-jpeg", "-r", "110", str(path), str(prefix)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=360,
            )
            if completed.returncode == 0:
                ocr_text = "\n".join(_ocr_image(image) for image in sorted(Path(temp_dir).glob("page-*.jpg")))
                text = f"{text}\n{ocr_text}"
    elif require_tools:
        findings.append(Finding("pdf-ocr-tool-unavailable", path.name))

    return f"{metadata}\n{text}", findings


def omit_assessment_pages_from_pdf(path: Path) -> int:
    """Remove PDF pages containing assignment or answer-key logistics.

    Returns the number of omitted pages. Raises ValueError when the deck cannot
    be sanitized without removing every page.
    """

    required = ("pdfinfo", "pdftotext", "pdfseparate", "pdfunite")
    if any(not shutil.which(command) for command in required):
        raise ValueError("PDF page-filtering tools are unavailable.")
    info = _run_text_command(["pdfinfo", str(path)])
    match = re.search(r"(?m)^Pages:\s*(\d+)\s*$", info)
    if not match:
        raise ValueError("PDF page count is unavailable.")
    page_count = int(match.group(1))
    omitted: set[int] = set()
    for page_number in range(1, page_count + 1):
        page_text = _run_text_command(
            [
                "pdftotext",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-layout",
                str(path),
                "-",
            ]
        )
        if ASSESSMENT_PAGE_RE.search(page_text):
            omitted.add(page_number)
    if not omitted:
        return 0
    if len(omitted) == page_count:
        raise ValueError("Every PDF page contains restricted assessment material.")

    with tempfile.TemporaryDirectory(prefix="canvas-pdf-filter-") as temp_dir:
        split_pattern = str(Path(temp_dir) / "page-%d.pdf")
        separated = subprocess.run(
            ["pdfseparate", str(path), split_pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=360,
        )
        if separated.returncode != 0:
            raise ValueError("PDF pages could not be separated.")
        retained = [
            str(Path(temp_dir) / f"page-{page_number}.pdf")
            for page_number in range(1, page_count + 1)
            if page_number not in omitted
        ]
        filtered = Path(temp_dir) / "filtered.pdf"
        united = subprocess.run(
            ["pdfunite", *retained, str(filtered)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=360,
        )
        if united.returncode != 0 or not filtered.exists():
            raise ValueError("Filtered PDF pages could not be combined.")
        shutil.copy2(filtered, path)
    return len(omitted)


def _extract_office_text(path: Path, require_tools: bool) -> tuple[str, list[Finding]]:
    findings: list[Finding] = []
    parts: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                suffix = Path(info.filename).suffix.lower()
                if suffix in {".xml", ".rels"}:
                    raw_xml = archive.read(info).decode("utf-8", errors="replace")
                    parts.append(raw_xml)
                    parts.append(html.unescape(re.sub(r"<[^>]+>", " ", raw_xml)))
                elif suffix in IMAGE_SUFFIXES:
                    if not shutil.which("tesseract"):
                        if require_tools:
                            findings.append(Finding("office-ocr-tool-unavailable", path.name))
                        continue
                    with tempfile.TemporaryDirectory(prefix="canvas-office-ocr-") as temp_dir:
                        image_path = Path(temp_dir) / Path(info.filename).name
                        image_path.write_bytes(archive.read(info))
                        parts.append(_ocr_image(image_path))
    except (OSError, zipfile.BadZipFile):
        findings.append(Finding("unreadable-office-file", path.name))
    return "\n".join(parts), findings


def extract_file_text(path: Path, require_binary_tools: bool = True) -> tuple[str, list[Finding]]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES or path.name == ".nojekyll":
        return path.read_text(encoding="utf-8", errors="replace"), []
    if suffix == ".pdf":
        return _extract_pdf_text(path, require_binary_tools)
    if suffix in OFFICE_SUFFIXES:
        return _extract_office_text(path, require_binary_tools)
    if suffix in IMAGE_SUFFIXES:
        if not shutil.which("tesseract"):
            finding = Finding("image-ocr-tool-unavailable", path.name)
            return "", [finding] if require_binary_tools else []
        return _ocr_image(path), []
    if suffix == ".ico":
        return "", []
    return "", [Finding("unsupported-public-file-type", path.name)]


def scan_text(text: str, protected_terms: Sequence[str], path: str) -> list[Finding]:
    findings: list[Finding] = []
    checks = (
        ("email-address", EMAIL_RE),
        ("phone-number", PHONE_RE),
        ("private-canvas-url", CANVAS_URL_RE),
        ("private-course-path", PRIVATE_COURSE_PATH_RE),
        ("credential-pattern", CREDENTIAL_RE),
        ("redaction-marker", REDACTION_MARKER_RE),
    )
    for category, pattern in checks:
        if pattern.search(text):
            findings.append(Finding(category, path))

    folded = " ".join(text.split()).casefold()
    for term in normalize_protected_terms(protected_terms):
        if term.casefold() in folded:
            findings.append(Finding("protected-student-or-course-identifier", path))
            break
    return findings


def scan_file(
    path: Path,
    protected_terms: Sequence[str],
    display_path: str | None = None,
    require_binary_tools: bool = True,
) -> list[Finding]:
    label = display_path or path.name
    if path.is_symlink():
        return [Finding("symlink-not-allowed", label)]
    if RAW_LMS_FILENAME_RE.search(label):
        return [Finding("raw-lms-identifier-in-filename", label)]
    folded_label = label.casefold()
    if any(term.casefold() in folded_label for term in normalize_protected_terms(protected_terms)):
        return [Finding("protected-identifier-in-filename", label)]
    text, extraction_findings = extract_file_text(path, require_binary_tools=require_binary_tools)
    findings = [Finding(item.category, label) for item in extraction_findings]
    findings.extend(scan_text(text, protected_terms, label))
    return findings


def scan_tree(
    root: Path,
    protected_terms: Sequence[str],
    include_roots: Sequence[str] = ("assets", "docs", "review-materials", "source", "templates"),
    require_binary_tools: bool = True,
) -> list[Finding]:
    findings: list[Finding] = []
    for relative_root in include_roots:
        base = root / relative_root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(root).as_posix()
            findings.extend(
                scan_file(
                    path,
                    protected_terms,
                    display_path=relative,
                    require_binary_tools=require_binary_tools,
                )
            )
    return sorted(set(findings), key=lambda item: (item.path, item.category))
