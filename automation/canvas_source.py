from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .privacy import (
        html_to_text,
        normalize_protected_terms,
        omit_assessment_pages_from_pdf,
        redact_text,
        scan_file,
    )
except ImportError:  # Direct script execution from automation/.
    from privacy import (
        html_to_text,
        normalize_protected_terms,
        omit_assessment_pages_from_pdf,
        redact_text,
        scan_file,
    )


class CanvasError(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectionResult:
    fingerprint: str
    protected_terms: tuple[str, ...]
    page_count: int
    slide_count: int


class CanvasClient:
    def __init__(self, base_url: str, token: str, timeout: int = 45) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v1"
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "isye4031-public-course-adapter/1.0",
        }

    def _request(self, url: str) -> urllib.response.addinfourl:
        request = urllib.request.Request(url, headers=self._headers)
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise CanvasError(f"Canvas request failed with HTTP {exc.code}.") from None
        except urllib.error.URLError:
            raise CanvasError("Canvas request failed because the service was unreachable.") from None

    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode(params or {}, doseq=True)
        url = f"{self.api_base}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{query}"
        with self._request(url) as response:
            try:
                return json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise CanvasError("Canvas returned an unreadable response.") from None

    def paginate(self, path: str, params: Mapping[str, Any] | None = None) -> list[Any]:
        initial = dict(params or {})
        initial.setdefault("per_page", 100)
        query = urllib.parse.urlencode(initial, doseq=True)
        url: str | None = f"{self.api_base}/{path.lstrip('/')}?{query}"
        results: list[Any] = []
        pages = 0
        while url:
            pages += 1
            if pages > 100:
                raise CanvasError("Canvas pagination exceeded the safety limit.")
            with self._request(url) as response:
                try:
                    payload = json.loads(response.read().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise CanvasError("Canvas returned an unreadable paginated response.") from None
                if not isinstance(payload, list):
                    raise CanvasError("Canvas returned an unexpected paginated response.")
                results.extend(payload)
                url = _next_link(response.headers.get("Link", ""))
        return results

    def download(self, url: str, destination: Path, max_bytes: int) -> None:
        parsed = urllib.parse.urlparse(url)
        base_host = urllib.parse.urlparse(self.base_url).hostname
        if parsed.scheme != "https" or parsed.hostname != base_host:
            raise CanvasError("Canvas supplied an untrusted download URL.")
        # Canvas file URLs carry a short-lived verifier. Do not attach the API
        # credential: redirects may terminate on object storage outside Canvas.
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self._headers["User-Agent"]},
        )
        try:
            response_context = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise CanvasError(f"Canvas file download failed with HTTP {exc.code}.") from None
        except urllib.error.URLError:
            raise CanvasError("Canvas file download was unreachable.") from None
        with response_context as response, destination.open("wb") as output:
            expected = response.headers.get("Content-Length")
            if expected and int(expected) > max_bytes:
                raise CanvasError("A Canvas slide file exceeded the configured size limit.")
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise CanvasError("A Canvas slide file exceeded the configured size limit.")
                output.write(chunk)


def _next_link(header: str) -> str | None:
    for section in header.split(","):
        if 'rel="next"' not in section:
            continue
        match = re.search(r"<([^>]+)>", section)
        if match:
            return match.group(1)
    return None


def _safe_filename(value: str, fallback: str) -> str:
    name = Path(value).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    if not name:
        name = fallback
    return name[:160]


def _safe_slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return (slug or fallback)[:100]


def _roster_terms(users: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    values: list[object] = []
    for user in users:
        values.extend(
            [
                user.get("id"),
                user.get("name"),
                user.get("short_name"),
                user.get("sortable_name"),
                user.get("login_id"),
                user.get("email"),
                user.get("sis_user_id"),
            ]
        )
    return normalize_protected_terms(values)


def _write_sanitized_page(
    destination: Path,
    title: str,
    body: str,
    protected_terms: tuple[str, ...],
) -> str:
    visible = html_to_text(body)
    sanitized_title = redact_text(title, protected_terms)
    sanitized_body = redact_text(visible, protected_terms)
    content = f"# {sanitized_title}\n\n{sanitized_body.strip()}\n"
    destination.write_text(content, encoding="utf-8")
    return content


def collect_canvas_source(config: Mapping[str, Any], destination: Path) -> CollectionResult:
    token_env = str(config.get("token_env", "CANVAS_API_KEY"))
    token = os.environ.get(token_env, "")
    if not token:
        raise CanvasError(f"Canvas credential environment variable {token_env} is unavailable.")

    base_url = str(config["base_url"])
    course_id = int(config["course_id"])
    client = CanvasClient(base_url, token)
    course = client.get_json(
        f"courses/{course_id}",
        {"include[]": ["syllabus_body"]},
    )
    if int(course.get("id", -1)) != course_id:
        raise CanvasError("Canvas returned the wrong course.")

    roster = client.paginate(
        f"courses/{course_id}/users",
        {"enrollment_type[]": ["student"]},
    )
    if not roster:
        raise CanvasError("The student roster privacy check returned no records; publication is blocked.")

    protected_terms = normalize_protected_terms((*_roster_terms(roster), str(course_id)))
    page_include = re.compile(str(config["page_title_include"]))
    page_exclude = re.compile(str(config["page_title_exclude"]))
    slide_include = re.compile(str(config["slide_name_include"]))
    slide_exclude = re.compile(str(config["slide_name_exclude"]))
    max_pages = int(config.get("max_pages", 60))
    max_files = int(config.get("max_slide_files", 24))
    max_bytes = int(config.get("max_slide_bytes", 50 * 1024 * 1024))

    destination.mkdir(parents=True, exist_ok=True)
    pages_dir = destination / "pages"
    slides_dir = destination / "approved-slides"
    pages_dir.mkdir()
    slides_dir.mkdir()

    fingerprint = hashlib.sha256()
    page_records: list[tuple[str, str]] = []

    syllabus_body = str(course.get("syllabus_body") or "").strip()
    if syllabus_body:
        output = pages_dir / "000-course-overview.md"
        content = _write_sanitized_page(
            output,
            str(config.get("public_course_title", "Course overview")),
            syllabus_body,
            protected_terms,
        )
        page_records.append((str(config.get("public_course_title", "Course overview")), output.name))
        fingerprint.update(content.encode("utf-8"))

    modules = client.paginate(
        f"courses/{course_id}/modules",
        {"include[]": ["items"]},
    )
    module_lines = ["# Public-safe module outline", ""]
    for module in modules:
        if not module.get("published", True):
            continue
        module_name = redact_text(str(module.get("name", "Module")), protected_terms)
        item_titles: list[str] = []
        for item in module.get("items") or []:
            if not item.get("published", True):
                continue
            if str(item.get("type", "")) not in {"File", "Page"}:
                continue
            item_title = str(item.get("title", ""))
            if page_exclude.search(item_title) or slide_exclude.search(item_title):
                continue
            item_titles.append(redact_text(item_title, protected_terms))
        if not item_titles:
            continue
        module_lines.append(f"## {module_name}")
        module_lines.append("")
        module_lines.extend(f"- {title}" for title in item_titles)
        module_lines.append("")
    if len(module_lines) > 2:
        module_content = "\n".join(module_lines).strip() + "\n"
        module_output = pages_dir / "001-module-outline.md"
        module_output.write_text(module_content, encoding="utf-8")
        page_records.append(("Module outline", module_output.name))
        fingerprint.update(module_content.encode("utf-8"))

    pages = client.paginate(f"courses/{course_id}/pages", {"published": "true"})
    selected_pages = []
    for page in pages:
        title = str(page.get("title", ""))
        if not page.get("published", True):
            continue
        if not page_include.search(title) or page_exclude.search(title):
            continue
        selected_pages.append(page)
    selected_pages = selected_pages[:max_pages]

    for index, page in enumerate(selected_pages, start=2):
        page_url = urllib.parse.quote(str(page.get("url", "")), safe="")
        if not page_url:
            continue
        detail = client.get_json(f"courses/{course_id}/pages/{page_url}")
        if not detail.get("published", True):
            continue
        title = str(detail.get("title", page.get("title", "Untitled page")))
        slug = _safe_slug(title, f"page-{index}")
        output = pages_dir / f"{index:03d}-{slug}.md"
        content = _write_sanitized_page(output, title, str(detail.get("body", "")), protected_terms)
        page_records.append((title, output.name))
        fingerprint.update(content.encode("utf-8"))

    slide_records: list[str] = []
    files = client.paginate(f"courses/{course_id}/files")
    selected_files: list[Mapping[str, Any]] = []
    for file_info in files:
        display_name = str(file_info.get("display_name") or file_info.get("filename") or "")
        suffix = Path(display_name).suffix.lower()
        content_type = str(file_info.get("content-type") or file_info.get("content_type") or "")
        is_slide_type = suffix in {".pdf", ".pptx"} or content_type in {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        if not is_slide_type:
            continue
        if not slide_include.search(display_name) or slide_exclude.search(display_name):
            continue
        if file_info.get("locked_for_user") or file_info.get("hidden_for_user"):
            continue
        selected_files.append(file_info)
    selected_files.sort(key=lambda item: (str(item.get("updated_at", "")), str(item.get("id", ""))))
    selected_files = selected_files[-max_files:]

    for file_info in selected_files:
        original = str(file_info.get("display_name") or file_info.get("filename") or "slide.pdf")
        file_id = str(file_info.get("id", "file"))
        safe_name = _safe_filename(original, f"slide-{file_id}.pdf")
        temporary_output = slides_dir / (
            f".download-{len(slide_records):03d}{Path(safe_name).suffix.lower()}"
        )
        url = str(file_info.get("url", ""))
        if not url:
            continue
        client.download(url, temporary_output, max_bytes=max_bytes)
        if temporary_output.suffix.lower() == ".pdf":
            try:
                omit_assessment_pages_from_pdf(temporary_output)
            except ValueError:
                temporary_output.unlink(missing_ok=True)
                continue
        findings = scan_file(
            temporary_output,
            protected_terms,
            display_path=safe_name,
            require_binary_tools=True,
        )
        if findings:
            temporary_output.unlink(missing_ok=True)
            continue
        content_digest = hashlib.sha256(temporary_output.read_bytes()).hexdigest()[:12]
        output_name = safe_name
        if (slides_dir / output_name).exists():
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix.lower()
            output_name = f"{stem}-{content_digest}{suffix}"
        output = slides_dir / output_name
        temporary_output.replace(output)
        slide_records.append(output_name)
        fingerprint.update(file_id.encode("utf-8"))
        fingerprint.update(str(file_info.get("updated_at", "")).encode("utf-8"))
        fingerprint.update(content_digest.encode("ascii"))

    public_title = str(config.get("public_course_title", "ISYE 4031 Regression and Forecasting"))
    index_lines = [
        f"# Sanitized source digest for {public_title}",
        "",
        "This material was selected from instructor-controlled course content and sanitized before agent access.",
        "Redaction markers are omissions and must never be copied into the public site.",
        "",
        "## Selected public-safe page digests",
        "",
    ]
    for title, filename in page_records:
        index_lines.append(f"- {redact_text(title, protected_terms)}: `pages/{filename}`")
    index_lines.extend(["", "## Approved English lecture slides", ""])
    for filename in slide_records:
        index_lines.append(f"- `approved-slides/{filename}`")
    index_lines.append("")
    manifest_text = "\n".join(index_lines)
    (destination / "README.md").write_text(manifest_text, encoding="utf-8")
    fingerprint.update(manifest_text.encode("utf-8"))

    return CollectionResult(
        fingerprint=fingerprint.hexdigest(),
        protected_terms=protected_terms,
        page_count=len(page_records),
        slide_count=len(slide_records),
    )
