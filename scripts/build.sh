#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$project_root/source"
docs_dir="$project_root/docs"

mkdir -p "$docs_dir/assets" "$docs_dir/materials" "$docs_dir/zh"
find "$docs_dir" -maxdepth 1 -type f -name '*.html' -delete
find "$docs_dir/zh" -maxdepth 1 -type f -name '*.html' -delete
find "$docs_dir/materials" -maxdepth 1 -type f -delete

for source_file in "$source_dir"/*.md; do
  page_name="$(basename "$source_file" .md)"
  pandoc "$source_file" \
    --from=markdown+tex_math_dollars+tex_math_single_backslash-native_divs \
    --to=html5 \
    --standalone \
    --mathjax \
    --template "$project_root/templates/page.html" \
    --output "$docs_dir/$page_name.html"
done

for source_file in "$source_dir"/zh/*.md; do
  page_name="$(basename "$source_file" .md)"
  pandoc "$source_file" \
    --from=markdown+tex_math_dollars+tex_math_single_backslash-native_divs \
    --to=html5 \
    --standalone \
    --mathjax \
    --template "$project_root/templates/page-zh.html" \
    --output "$docs_dir/zh/$page_name.html"
done

cp "$project_root/assets/style.css" "$docs_dir/assets/style.css"
while IFS= read -r -d '' material; do
  cp "$material" "$docs_dir/materials/$(basename "$material")"
done < <(
  find "$project_root/review-materials" -maxdepth 1 -type f \
    \( -iname '*.pdf' -o -iname '*.pptx' \) -print0
)
touch "$docs_dir/.nojekyll"

echo "Built public course draft in $docs_dir"
