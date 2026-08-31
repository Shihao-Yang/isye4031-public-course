#!/usr/bin/env bash
set -euo pipefail

umask 077
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

canvas_token="$(security find-generic-password -a "$USER" -s "Codex Canvas API" -w 2>/dev/null || true)"
if [[ -z "$canvas_token" ]]; then
  echo "Canvas credential is unavailable in macOS Keychain." >&2
  exit 1
fi
export CANVAS_API_KEY="$canvas_token"
unset canvas_token

mkdir -p "$project_root/automation/logs"
chmod 700 "$project_root/automation/logs"

exec python3 "$project_root/automation/pipeline.py" \
  --config "$project_root/automation/config.local.toml"
