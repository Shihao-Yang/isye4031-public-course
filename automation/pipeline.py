from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .canvas_source import CanvasClient, CanvasError, collect_canvas_source
    from .privacy import Finding, scan_tree
except ImportError:  # Direct script execution from automation/.
    from canvas_source import CanvasClient, CanvasError, collect_canvas_source
    from privacy import Finding, scan_tree


class WorkflowError(RuntimeError):
    pass


TOKENISH_RE = re.compile(r"(?i)(?:\b\d+~[A-Za-z0-9_-]{20,}\b|verifier=[^&\s]+|access_token=[^&\s]+)")
ALLOWED_CHANGED_ROOTS = {"docs", "review-materials", "source"}


def _safe_message(value: str) -> str:
    return TOKENISH_RE.sub("[redacted]", value).strip()


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout: int = 900,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env.setdefault("GIT_TERMINAL_PROMPT", "0")
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=merged_env,
        )
    except subprocess.TimeoutExpired:
        raise WorkflowError(f"Command timed out: {Path(command[0]).name}") from None
    if completed.returncode != 0:
        detail = _safe_message(completed.stderr.splitlines()[-1] if completed.stderr else "")
        suffix = f" ({detail})" if detail else ""
        raise WorkflowError(f"Command failed: {Path(command[0]).name}{suffix}")
    return completed


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise WorkflowError(f"Unable to read configuration: {exc}") from None
    for section in ("canvas", "site", "agent", "schedule"):
        if section not in config:
            raise WorkflowError(f"Configuration section [{section}] is missing.")
    return config


def _codex_command(config: Mapping[str, Any]) -> str:
    configured = str(config.get("command", "codex"))
    resolved = shutil.which(configured) if not Path(configured).is_absolute() else configured
    if not resolved or not Path(resolved).exists():
        raise WorkflowError("Codex CLI is unavailable.")
    return str(resolved)


def _validate_canvas(config: Mapping[str, Any]) -> None:
    token_env = str(config.get("token_env", "CANVAS_API_KEY"))
    token = os.environ.get(token_env, "")
    if not token:
        raise WorkflowError(f"Canvas credential environment variable {token_env} is unavailable.")
    client = CanvasClient(str(config["base_url"]), token)
    profile = client.get_json("users/self/profile")
    course = client.get_json(f"courses/{int(config['course_id'])}")
    if not profile.get("id"):
        raise WorkflowError("Canvas profile validation failed.")
    if int(course.get("id", -1)) != int(config["course_id"]):
        raise WorkflowError("Canvas course validation failed.")


def _validate_github_push(repository: str, branch: str) -> None:
    with tempfile.TemporaryDirectory(prefix="canvas-git-preflight-") as temp_dir:
        clone = Path(temp_dir) / "site"
        _run(["git", "clone", "--quiet", "--depth", "1", "--branch", branch, repository, str(clone)])
        _run(["git", "push", "--dry-run", "origin", f"HEAD:{branch}"], cwd=clone, timeout=120)


def preflight(config: Mapping[str, Any], *, require_push: bool) -> None:
    for command in (
        "git",
        "python3",
        "pdftotext",
        "pdfinfo",
        "pdfseparate",
        "pdfunite",
        "pdftoppm",
        "tesseract",
        "pandoc",
    ):
        if not shutil.which(command):
            raise WorkflowError(f"Required command is unavailable: {command}")
    _codex_command(config["agent"])
    _validate_canvas(config["canvas"])
    if require_push:
        _validate_github_push(str(config["site"]["repository"]), str(config["site"]["branch"]))


def _state_path(project_root: Path) -> Path:
    return project_root / "automation" / "state.json"


def _read_previous_fingerprint(project_root: Path) -> str | None:
    path = _state_path(project_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("fingerprint") if payload.get("version") == 1 else None


def _write_state(project_root: Path, fingerprint: str) -> None:
    payload = {
        "version": 1,
        "fingerprint": fingerprint,
        "last_success_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    }
    path = _state_path(project_root)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _clone_site(repository: str, branch: str, destination: Path) -> None:
    _run(
        ["git", "clone", "--quiet", "--depth", "1", "--branch", branch, repository, str(destination)],
        timeout=180,
    )


def _replace_canvas_slides(source_dir: Path, site_dir: Path) -> None:
    review_dir = site_dir / "review-materials"
    review_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("canvas-*", "public-slide-*"):
        for old_file in review_dir.glob(pattern):
            if old_file.is_file():
                old_file.unlink()
    for slide in sorted((source_dir / "approved-slides").glob("*")):
        if slide.is_file():
            shutil.copy2(slide, review_dir / slide.name)


def _run_adaptation_agent(
    project_root: Path,
    work_root: Path,
    codex_command: str,
) -> None:
    prompt = (project_root / "automation" / "prompts" / "adapt.md").read_text(encoding="utf-8")
    last_message = work_root / "agent-last-message.txt"
    _run(
        [
            codex_command,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--approve-for-me",
            "--skip-git-repo-check",
            "--cd",
            str(work_root),
            "--output-last-message",
            str(last_message),
            prompt,
        ],
        cwd=work_root,
        timeout=1800,
        env={"CODEX_CI": "1"},
    )


def _changed_paths(site_dir: Path) -> set[str]:
    tracked = _run(["git", "diff", "--name-only"], cwd=site_dir).stdout.splitlines()
    untracked = _run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=site_dir
    ).stdout.splitlines()
    return {path.strip() for path in (*tracked, *untracked) if path.strip()}


def _enforce_change_scope(site_dir: Path) -> set[str]:
    changed = _changed_paths(site_dir)
    outside = sorted(path for path in changed if Path(path).parts[0] not in ALLOWED_CHANGED_ROOTS)
    if outside:
        raise WorkflowError("Agent or build changed a path outside the public-content allowlist.")
    return changed


def _run_site_build(site_dir: Path) -> None:
    _run(["bash", "scripts/build.sh"], cwd=site_dir, timeout=600)
    _run(["bash", "scripts/check.sh"], cwd=site_dir, timeout=600)


def _report_findings(findings: Sequence[Finding]) -> str:
    summary = sorted({f"{finding.category} in {finding.path}" for finding in findings})
    return "; ".join(summary[:12])


def _run_privacy_audit_agent(
    project_root: Path,
    site_dir: Path,
    work_root: Path,
    codex_command: str,
) -> None:
    prompt = (project_root / "automation" / "prompts" / "audit.md").read_text(encoding="utf-8")
    schema = project_root / "automation" / "privacy_audit_schema.json"
    output = work_root / "privacy-audit.json"
    _run(
        [
            codex_command,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--sandbox",
            "read-only",
            "--cd",
            str(site_dir),
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(output),
            prompt,
        ],
        cwd=site_dir,
        timeout=1200,
        env={"CODEX_CI": "1"},
    )
    try:
        audit = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise WorkflowError("Codex privacy audit returned an unreadable result.") from None
    if audit.get("pass") is not True or audit.get("findings"):
        finding_labels = sorted(
            {
                f"{item.get('category', 'privacy-finding')} in {item.get('path', 'unknown path')}"
                for item in audit.get("findings", [])
                if isinstance(item, dict)
            }
        )
        detail = "; ".join(finding_labels[:8]) or "unspecified privacy finding"
        raise WorkflowError(f"Codex privacy audit blocked publication: {detail}")


def _commit_and_push(site_dir: Path, branch: str) -> None:
    _run(["git", "add", "--", "source", "review-materials", "docs"], cwd=site_dir)
    staged = _run(["git", "diff", "--cached", "--name-only"], cwd=site_dir).stdout.strip()
    if not staged:
        return
    date = dt.datetime.now().astimezone().date().isoformat()
    _run(
        [
            "git",
            "-c",
            "user.name=Shihao Yang",
            "-c",
            "user.email=shihao-yang@users.noreply.github.com",
            "commit",
            "--quiet",
            "-m",
            f"Update public course materials from Canvas ({date})",
        ],
        cwd=site_dir,
    )
    _run(["git", "push", "origin", f"HEAD:{branch}"], cwd=site_dir, timeout=300)


def run_pipeline(
    project_root: Path,
    config: Mapping[str, Any],
    *,
    dry_run: bool,
    force: bool,
) -> None:
    canvas_config = config["canvas"]
    site_config = config["site"]
    agent_config = config["agent"]
    publish = bool(site_config.get("publish", True)) and not dry_run

    print("Preflight: validating local tools, Canvas read access, and publication access.")
    preflight(config, require_push=publish)

    with tempfile.TemporaryDirectory(prefix="canvas-publication-") as temp_dir:
        work_root = Path(temp_dir)
        source_dir = work_root / "canvas-source"
        site_dir = work_root / "site"

        print("Canvas: collecting the allowlisted course pages and lecture slides.")
        collection = collect_canvas_source(canvas_config, source_dir)
        if collection.page_count == 0 and collection.slide_count == 0:
            raise WorkflowError("Canvas selection produced no public-facing source material.")
        print(
            f"Canvas: sanitized {collection.page_count} page digests and approved "
            f"{collection.slide_count} lecture slide files."
        )

        if not force and _read_previous_fingerprint(project_root) == collection.fingerprint:
            print("No Canvas source changes were detected; no site or GitHub changes are needed.")
            return

        print("Site: preparing a clean copy of the public GitHub Pages repository.")
        _clone_site(str(site_config["repository"]), str(site_config["branch"]), site_dir)
        _replace_canvas_slides(source_dir, site_dir)

        codex_command = _codex_command(agent_config)
        if bool(agent_config.get("adapt", True)):
            print("Codex: adapting the sanitized material into the bilingual public site.")
            _run_adaptation_agent(project_root, work_root, codex_command)

        _enforce_change_scope(site_dir)
        print("Site: building English and Chinese pages and running deterministic checks.")
        _run_site_build(site_dir)
        changed = _enforce_change_scope(site_dir)

        print("Privacy: scanning text, metadata, slide text, and OCR output.")
        findings = scan_tree(site_dir, collection.protected_terms, require_binary_tools=True)
        if findings:
            raise WorkflowError(f"Deterministic privacy gate blocked publication: {_report_findings(findings)}")

        if bool(agent_config.get("privacy_audit", True)):
            print("Codex: running a separate read-only publication-safety audit.")
            _run_privacy_audit_agent(project_root, site_dir, work_root, codex_command)

        if not changed:
            print("The public site already matches the sanitized Canvas material.")
            if not dry_run:
                _write_state(project_root, collection.fingerprint)
            return

        if dry_run:
            print(f"Dry run passed with {len(changed)} changed public-site files; nothing was committed or pushed.")
            return

        if not publish:
            print("Publication is disabled in configuration; verified changes were not pushed.")
            _write_state(project_root, collection.fingerprint)
            return

        print("GitHub: committing and pushing the verified public-site update.")
        _commit_and_push(site_dir, str(site_config["branch"]))
        _write_state(project_root, collection.fingerprint)
        print("Publication completed successfully.")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt allowlisted Canvas teaching material into the public ISYE 4031 site."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Build and audit without committing or pushing.")
    parser.add_argument("--force", action="store_true", help="Run even when the Canvas source fingerprint is unchanged.")
    parser.add_argument("--preflight", action="store_true", help="Validate dependencies and access, then exit.")
    parser.add_argument(
        "--skip-push-check",
        action="store_true",
        help="With --preflight, do not require GitHub write authentication.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    project_root = args.config.resolve().parents[1]
    try:
        config = load_config(args.config)
        if args.preflight:
            preflight(config, require_push=not args.skip_push_check)
            print("Preflight passed.")
            return 0
        run_pipeline(project_root, config, dry_run=args.dry_run, force=args.force)
        return 0
    except (CanvasError, WorkflowError) as exc:
        print(f"Publication workflow stopped safely: {_safe_message(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
