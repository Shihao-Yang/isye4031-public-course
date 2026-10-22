#!/usr/bin/env python3
"""Build a course-site preview or release with explicitly approved replays."""
from pathlib import Path
import argparse
import re
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--release', action='store_true', help='Stage the owner-approved replay in docs for publication')
args = parser.parse_args()
site = Path(__file__).resolve().parents[1]
player = site.parents[1] / 'llm-chat-replay'
subprocess.run(['npm', 'run', 'build', '--', '--base=./'], cwd=player, check=True, stdout=subprocess.DEVNULL)
subprocess.run(['bash', 'scripts/build.sh'], cwd=site, check=True)
preview = site / 'docs' if args.release else site / 'tmp/replay-preview/isye4031-public-course'
if not args.release:
    shutil.copytree(site / 'docs', preview, dirs_exist_ok=True)
replay = preview / 'replay'
replay.mkdir(parents=True, exist_ok=True)
shutil.copy2(player / 'dist/index.html', replay / 'index.html')
if (replay / 'assets').exists():
    shutil.rmtree(replay / 'assets')
shutil.copytree(player / 'dist/assets', replay / 'assets', dirs_exist_ok=True)
(replay / 'examples').mkdir(exist_ok=True)
# Keep this explicit list aligned with publication-manifest.yml.
for lesson in ('lecture12', 'lecture13', 'lecture13-v2', 'lecture13-v3', 'lecture14-v2'):
    transcript = player / f'public/examples/{lesson}-replay.md'
    text = transcript.read_text()
    paths = set(re.findall(rf'\]\(/(examples/{re.escape(lesson)}/[^)]+)\)', text))
    if lesson == 'lecture14-v2':
        paths.update(f'examples/{lesson}/{name}' for name in (
            'constructed_sales_seed4031.csv', 'fixed_sales_split.csv',
            'sales_generation_metadata.json', 'requirements.txt'))
    else:
        paths.add(f'examples/{lesson}/sales_territory.csv')
    shutil.copy2(transcript, replay / f'examples/{lesson}-replay.md')
    asset_root = (player / f'public/examples/{lesson}').resolve()
    for path in sorted(paths):
        source = (player / 'public' / path).resolve()
        if not source.is_relative_to(asset_root) or source.suffix not in {'.py', '.csv', '.json', '.png', '.txt', '.md'}:
            raise SystemExit(f'Unapproved replay asset path: {path}')
        target = replay / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
shutil.copy2(player / 'LICENSE', replay / 'LICENSE')
print(f'Local preview: {preview}')
print('Release payload staged.' if args.release else 'Local preview staged. No Git operations performed.')
