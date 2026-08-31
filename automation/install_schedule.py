from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
import tomllib
from pathlib import Path


LABEL = "com.shihaoyang.canvas-to-github"


def _run(command: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> int:
    completed = subprocess.run(command, check=False, env=env)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed: {Path(command[0]).name}")
    return completed.returncode


def _token_from_keychain() -> str:
    completed = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-a",
            os.environ.get("USER", ""),
            "-s",
            "Codex Canvas API",
            "-w",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    token = completed.stdout.strip()
    if completed.returncode != 0 or not token:
        raise RuntimeError("Canvas API credential is unavailable in macOS Keychain.")
    return token


def _load_schedule(config_path: Path) -> tuple[int, int, list[int], str]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    schedule = config["schedule"]
    hour = int(schedule["hour"])
    minute = int(schedule["minute"])
    weekdays = [int(value) for value in schedule["weekdays"]]
    timezone = str(schedule.get("timezone", "America/New_York"))
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise RuntimeError("The configured schedule time is invalid.")
    if not weekdays or any(day < 0 or day > 7 for day in weekdays):
        raise RuntimeError("The configured launchd weekdays are invalid.")
    return hour, minute, weekdays, timezone


def _preflight(project_root: Path, config_path: Path) -> None:
    environment = os.environ.copy()
    environment["CANVAS_API_KEY"] = _token_from_keychain()
    try:
        _run(
            [
                sys.executable,
                str(project_root / "automation" / "pipeline.py"),
                "--config",
                str(config_path),
                "--preflight",
            ],
            env=environment,
        )
    finally:
        environment.pop("CANVAS_API_KEY", None)


def install(project_root: Path, config_path: Path) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("This scheduler installer requires macOS launchd.")
    hour, minute, weekdays, timezone = _load_schedule(config_path)
    if timezone != "America/New_York":
        raise RuntimeError("launchd uses the Mac's local timezone; this workflow expects America/New_York.")

    _preflight(project_root, config_path)

    logs = project_root / "automation" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logs.chmod(0o700)
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    plist_path = launch_agents / f"{LABEL}.plist"

    payload = {
        "Label": LABEL,
        "ProgramArguments": ["/bin/bash", str(project_root / "automation" / "run-scheduled.sh")],
        "WorkingDirectory": str(project_root),
        "StartCalendarInterval": [
            {"Weekday": weekday, "Hour": hour, "Minute": minute} for weekday in weekdays
        ],
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "StandardOutPath": str(logs / "launchd.stdout.log"),
        "StandardErrorPath": str(logs / "launchd.stderr.log"),
        "EnvironmentVariables": {
            "HOME": str(Path.home()),
            "USER": os.environ.get("USER", ""),
            "PATH": "/opt/homebrew/bin:/usr/local/bin:"
            + str(Path.home() / ".local" / "bin")
            + ":/usr/bin:/bin:/usr/sbin:/sbin",
        },
    }

    with plist_path.open("wb") as output:
        plistlib.dump(payload, output, sort_keys=False)
    plist_path.chmod(0o600)

    domain = f"gui/{os.getuid()}"
    _run(["launchctl", "bootout", domain, str(plist_path)], check=False)
    _run(["launchctl", "bootstrap", domain, str(plist_path)])
    print(f"Installed {LABEL}: Tuesday and Thursday at {hour:02d}:{minute:02d} local time.")


def uninstall() -> None:
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    domain = f"gui/{os.getuid()}"
    if plist_path.exists():
        _run(["launchctl", "bootout", domain, str(plist_path)], check=False)
        plist_path.unlink()
    print(f"Uninstalled {LABEL}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the Canvas-to-GitHub launchd schedule.")
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    try:
        if args.uninstall:
            uninstall()
            return 0
        project_root = Path(__file__).resolve().parents[1]
        install(project_root, project_root / "automation" / "config.local.toml")
        return 0
    except (OSError, RuntimeError, KeyError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"Schedule setup stopped safely: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
