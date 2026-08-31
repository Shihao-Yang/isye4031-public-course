#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docs_dir="$project_root/docs"

test -f "$docs_dir/index.html"
test -f "$docs_dir/zh/index.html"

if rg -n '—' "$project_root/source" "$project_root/templates" "$project_root/assets"; then
  echo "Em dash found in public source." >&2
  exit 1
fi

if rg -n -i '@gatech\.edu|verifier=|authorization:[[:space:]]*bearer|learning[_ -]?management[_ -]?system[_ -]?(api[_ -]?)?token|answer key' "$docs_dir" --glob '!materials/**'; then
  echo "Potentially restricted pattern found in generated pages." >&2
  exit 1
fi

while IFS= read -r href; do
  case "$href" in
    http://*|https://*|mailto:*|\#*) continue ;;
  esac
  href_path="${href%%#*}"
  test -e "$docs_dir/$href_path" || {
    echo "Broken local link: $href" >&2
    exit 1
  }
done < <(rg -o 'href="[^"]+"' "$docs_dir"/*.html | sed -E 's/.*href="([^"]+)"/\1/' | sort -u)

while IFS= read -r href; do
  case "$href" in
    http://*|https://*|mailto:*|\#*) continue ;;
  esac
  href_path="${href%%#*}"
  test -e "$docs_dir/zh/$href_path" || {
    echo "Broken Chinese local link: $href" >&2
    exit 1
  }
done < <(rg -o 'href="[^"]+"' "$docs_dir/zh"/*.html | sed -E 's/.*href="([^"]+)"/\1/' | sort -u)

for page in "$docs_dir"/*.html; do
  test "$(rg -c '<h1' "$page")" -eq 1 || {
    echo "Expected exactly one h1 in $page" >&2
    exit 1
  }
done


for page in "$docs_dir/zh"/*.html; do
  test "$(rg -c '<h1' "$page")" -eq 1 || {
    echo "Expected exactly one h1 in $page" >&2
    exit 1
  }
done

echo "Public course checks passed."
